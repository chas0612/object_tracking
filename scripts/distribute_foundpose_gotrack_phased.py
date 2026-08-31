#!/usr/bin/env python3
"""Run FoundPose + GoTrack as durable, separately launchable phases.

Unlike ``distribute_foundpose_gotrack.py``, a worker does not finish one
episode end-to-end.  It drains one phase at a time:

    mask -> foundpose -> gotrack -> debug

The mask phase saves only undistorted init-frame PNGs. GoTrack decodes raw
video and remaps frames in memory, so no full undistorted AVI is created. The
SAM3 image model stays resident for the complete mask phase. During the
FoundPose phase, remaining objects are greedily balanced by episode count and
each object stays on one worker. Tasks are ordered by episode, so each worker
reuses one FoundPose model, object representation, and silhouette renderer
across all episodes of that object. GoTrack remains
process-isolated per episode because its model startup is small and its worker
pools benefit from a clean lifecycle.

This script uses its own schedule schema and never mutates legacy schedules.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import distribute_foundpose_gotrack as legacy


PHASES = ("mask", "foundpose", "gotrack", "debug")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _schedule(shared: Path, args: argparse.Namespace) -> Path:
    return shared / args.runs_root_rel / args.schedule_id


def _task_path(schedule: Path, task_id: str) -> Path:
    return schedule / "tasks" / f"{task_id}.json"


def _paths(shared: Path, schedule: Path, task: dict[str, Any]) -> dict[str, Path]:
    episode = shared / task["episode_rel"]
    attempt = episode / "object_tracking_foundpose_gotrack" / schedule.name / "attempt_01"
    frame_index = int(task["init_frame_index"])
    return {
        "episode": episode,
        "mesh": shared / task["mesh_rel"],
        "assets": shared / task["assets_rel"],
        "attempt": attempt,
        "frame": attempt / f"foundpose_frame_{frame_index:06d}",
        "init": attempt / "foundpose_init",
        "track": attempt / "gotrack_tracking",
    }


def _phase_ready(task: dict[str, Any], phase: str) -> bool:
    index = PHASES.index(phase)
    return all(task["phases"][name]["status"] == "completed" for name in PHASES[:index])


def _object_shard(object_name: str, count: int) -> int:
    digest = hashlib.sha256(object_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _foundpose_assignment_path(schedule: Path) -> Path:
    return schedule / "assignments" / "foundpose.json"


def _prepare_foundpose_assignment(
    schedule: Path,
    worker_count: int,
    retry_failed: bool,
    max_attempts: int,
) -> dict[str, int]:
    """Balance remaining episodes while keeping every object on one worker."""
    weights: dict[str, int] = {}
    for path in (schedule / "tasks").glob("*.json"):
        task = _read(path)
        if not _phase_ready(task, "foundpose"):
            continue
        state = task["phases"]["foundpose"]
        runnable = state["status"] == "pending" or (
            retry_failed
            and state["status"] == "failed"
            and int(state["attempts"]) < max_attempts
        )
        if runnable:
            name = task["object_name"]
            weights[name] = weights.get(name, 0) + 1

    loads = [0] * worker_count
    assignments: dict[str, int] = {}
    for object_name, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
        rank = min(range(worker_count), key=lambda candidate: (loads[candidate], candidate))
        assignments[object_name] = rank
        loads[rank] += weight

    _write(_foundpose_assignment_path(schedule), {
        "worker_count": worker_count,
        "loads": loads,
        "object_assignments": assignments,
        "created_utc": legacy._now(),
    })
    print(
        "[foundpose-balance] "
        + " ".join(f"rank{rank}={load}" for rank, load in enumerate(loads)),
        flush=True,
    )
    return assignments


def _load_foundpose_assignment(schedule: Path, worker_count: int) -> dict[str, int] | None:
    path = _foundpose_assignment_path(schedule)
    if not path.is_file():
        return None
    value = _read(path)
    if int(value.get("worker_count", -1)) != worker_count:
        return None
    return {str(name): int(rank) for name, rank in value["object_assignments"].items()}


def _claim(
    schedule: Path,
    phase: str,
    worker_id: str,
    retry_failed: bool,
    max_attempts: int,
    worker_rank: int,
    worker_count: int,
    foundpose_assignments: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    candidates = []
    for path in (schedule / "tasks").glob("*.json"):
        task = _read(path)
        if not _phase_ready(task, phase):
            continue
        if phase == "foundpose":
            owner = (
                foundpose_assignments.get(task["object_name"])
                if foundpose_assignments is not None
                else _object_shard(task["object_name"], worker_count)
            )
            if owner != worker_rank:
                continue
        state = task["phases"][phase]
        allowed = state["status"] == "pending" or (
            retry_failed and state["status"] == "failed" and int(state["attempts"]) < max_attempts
        )
        if allowed:
            candidates.append((task["object_name"], int(task["episode"]), path, task))
    for _, _, path, task in sorted(candidates):
        state = task["phases"][phase]
        claim = schedule / "claims" / phase / f"{task['task_id']}.lock"
        if state["status"] == "failed" and claim.exists():
            shutil.rmtree(claim, ignore_errors=True)
        try:
            claim.mkdir(parents=True)
        except FileExistsError:
            continue
        state.update({
            "status": "running",
            "attempts": int(state["attempts"]) + 1,
            "worker_id": worker_id,
            "started_utc": legacy._now(),
            "updated_utc": legacy._now(),
        })
        task["active_phase"] = phase
        task["updated_utc"] = legacy._now()
        _write(path, task)
        _write(claim / "claim.json", {
            "worker_id": worker_id,
            "host": socket.gethostname(),
            "phase": phase,
            "claimed_utc": legacy._now(),
        })
        return task
    return None


def _subprocess(command: list[str], log: Any) -> None:
    log.write("$ " + " ".join(shlex.quote(value) for value in command) + "\n")
    log.flush()
    subprocess.run(
        command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
        text=True, check=True,
    )


def _foundpose_namespace(manifest: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{
        "foundpose_selection_mode": manifest["foundpose_selection_mode"],
        "foundpose_candidate_rank": manifest["foundpose_candidate_rank"],
        "foundpose_per_view_candidates": manifest["foundpose_per_view_candidates"],
        "foundpose_global_rotation_count": manifest["foundpose_global_rotation_count"],
        "foundpose_global_coarse_max_side": manifest["foundpose_global_coarse_max_side"],
        "foundpose_global_refine_top_k": manifest["foundpose_global_refine_top_k"],
        "foundpose_global_min_rotation_separation_deg": manifest["foundpose_global_min_rotation_separation_deg"],
        "foundpose_global_asymmetry_refine_top_k": manifest["foundpose_global_asymmetry_refine_top_k"],
        "foundpose_global_asymmetry_max_side": manifest["foundpose_global_asymmetry_max_side"],
        "foundpose_global_asymmetry_score_margin": manifest["foundpose_global_asymmetry_score_margin"],
        "foundpose_global_asymmetry_weight": manifest["foundpose_global_asymmetry_weight"],
        "foundpose_global_asymmetry_force": manifest["foundpose_global_asymmetry_force"],
        "foundpose_global_dino_rerank": manifest["foundpose_global_dino_rerank"],
        "foundpose_global_dino_score_margin": manifest["foundpose_global_dino_score_margin"],
        "foundpose_global_dino_inlier_threshold_px": manifest["foundpose_global_dino_inlier_threshold_px"],
    })


def _recovery_namespace(manifest: dict[str, Any]) -> SimpleNamespace:
    values = vars(_foundpose_namespace(manifest)).copy()
    for key in (
        "num_cameras", "camera_micro_batch_size", "max_video_duration_skew_sec",
        "max_frames", "max_trailing_missing_frames", "tail_recovery_frame_step",
        "tail_recovery_max_seed_attempts", "tail_recovery_min_mask_views",
        "tail_recovery_overlap_frames", "tail_recovery_min_connection_frames",
        "tail_recovery_min_suffix_coverage", "tail_recovery_max_translation_error_m",
        "tail_recovery_max_rotation_error_deg",
        "tail_recovery_max_rotation_alignment_dispersion_deg",
    ):
        values[key] = manifest[key]
    values["inline_gotrack_undistort"] = True
    return SimpleNamespace(**values)


def _run_phase_task(
    phase: str,
    task: dict[str, Any],
    shared: Path,
    schedule: Path,
    manifest: dict[str, Any],
    sam_segmentor: object | None,
) -> dict[str, Any]:
    paths = _paths(shared, schedule, task)
    paths["attempt"].mkdir(parents=True, exist_ok=True)
    state = task["phases"][phase]
    log_path = schedule / "logs" / phase / (
        f"{task['task_id']}.attempt{state['attempts']}.{state['worker_id']}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state["log"] = str(log_path)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(
                f"task={task['task_id']} phase={phase} worker={state['worker_id']} "
                f"started_utc={legacy._now()}\n"
            )
            if int(state["attempts"]) > 1 and phase in {"foundpose", "gotrack"}:
                retry_source = paths["init"] if phase == "foundpose" else paths["track"]
                if retry_source.exists() and not retry_source.is_symlink():
                    archive = (
                        paths["attempt"] / "phase_retry_archive"
                        / f"{phase}_attempt_{int(state['attempts']) - 1:02d}"
                    )
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    if archive.exists():
                        raise FileExistsError(f"Retry archive already exists: {archive}")
                    retry_source.rename(archive)
                    log.write(f"[retry-archive] {retry_source} -> {archive}\n")
                    log.flush()
            if phase == "mask":
                if sam_segmentor is None:
                    raise RuntimeError("mask phase has no persistent SAM3 model")
                from src.process.mask import process_episode_sam3_frame

                prompts = task.get("sam3_prompts") or [task["source_object"].replace("_", " ")]
                success = False
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    for prompt in prompts:
                        done, _, _ = process_episode_sam3_frame(
                            seg=sam_segmentor,
                            capture_dir=paths["episode"],
                            prompt=prompt,
                            frame_index=int(task["init_frame_index"]),
                            output_dir=paths["frame"],
                            video_dir=paths["episode"] / "videos",
                            undistort_frame=True,
                        )
                        task["sam3_prompt_used"] = prompt
                        if done >= int(manifest["min_sam3_mask_views"]):
                            success = True
                            break
                if not success:
                    raise RuntimeError(f"SAM3 masks below minimum for prompts={prompts}")
            elif phase == "foundpose":
                from src.process.foundpose_init_capture import main as foundpose_main

                command = legacy._foundpose_command(
                    [], capture=paths["episode"], frame_dir=paths["frame"],
                    mesh=paths["mesh"], object_name=task["object_name"],
                    assets=paths["assets"], output_dir=paths["init"],
                    args=_foundpose_namespace(manifest),
                )
                argv = command[1:] + ["--persistent-session"]
                log.write("$ in-process foundpose_init_capture.py " + " ".join(
                    shlex.quote(value) for value in argv
                ) + "\n")
                log.flush()
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    return_code = foundpose_main(argv)
                if return_code:
                    raise RuntimeError(f"FoundPose returned {return_code}")
            elif phase == "gotrack":
                _subprocess([
                    sys.executable, "src/process/gotrack_capture.py",
                    "--capture-dir", str(paths["episode"]),
                    "--video-dir", str(paths["episode"] / "videos"),
                    "--inline-undistort",
                    "--mesh", str(paths["mesh"]),
                    "--init-pose", str(paths["init"] / "init_pose_world.npy"),
                    "--object-name", task["object_name"],
                    "--num-cameras", str(manifest["num_cameras"]),
                    "--allow-fewer-cameras",
                    "--camera-micro-batch-size", str(manifest["camera_micro_batch_size"]),
                    "--init-frame-index", str(task["init_frame_index"]),
                    "--reverse-stop-frame-index", str(manifest["reverse_stop_frame_index"]),
                    "--max-video-duration-skew-sec", str(manifest["max_video_duration_skew_sec"]),
                    "--max-frames", str(manifest["max_frames"]),
                    "--output-dir", str(paths["track"]),
                ], log)
                summary = legacy._tracking_summary(
                    task, paths["attempt"], float(manifest["min_valid_pose_coverage"]),
                    int(manifest["max_trailing_missing_frames"]),
                    int(manifest["reverse_stop_frame_index"]),
                )
                task["tracking_summary"] = summary
                if (
                    not summary["complete"]
                    and manifest["tail_recovery"]
                    and int(summary.get("trailing_missing", 0))
                    > int(manifest["max_trailing_missing_frames"])
                ):
                    sam3_prefix = [
                        str(Path.home() / "anaconda3/bin/conda"), "run",
                        "--no-capture-output", "-n", manifest["sam3_env"],
                        "python", "-u",
                    ]
                    recovered = legacy._attempt_tail_recovery(
                        schedule_dir=schedule, task=task,
                        attempt_dir=paths["attempt"], episode=paths["episode"],
                        mesh=paths["mesh"], assets=paths["assets"],
                        track_dir=paths["track"], gotrack=[sys.executable],
                        sam3=sam3_prefix,
                        prompts=task.get("sam3_prompts") or [task["source_object"].replace("_", " ")],
                        args=_recovery_namespace(manifest), log=log,
                    )
                    task.setdefault("tail_recovery", {})["published"] = recovered
                    summary = legacy._tracking_summary(
                        task, paths["attempt"],
                        float(manifest["min_valid_pose_coverage"]),
                        int(manifest["max_trailing_missing_frames"]),
                        int(manifest["reverse_stop_frame_index"]),
                    )
                    task["tracking_summary"] = summary
                if not summary["complete"]:
                    raise RuntimeError(f"tracking incomplete: {summary['reason']}")
            elif phase == "debug":
                records = (
                    paths["track"] / "gotrack_output" / task["object_name"]
                    / "world_pose_records.json"
                )
                episode_sheet = paths["episode"] / f"gotrack_debug_sheet_{schedule.name}_attempt_01.jpg"
                central = (
                    shared / manifest["debug_sheet_output_root_rel"] / schedule.name
                    / f"{task['task_id']}.jpg"
                )
                if not episode_sheet.is_file():
                    _subprocess([
                        sys.executable, "src/process/render_gotrack_debug_sheet.py",
                        "--capture-dir", str(paths["episode"]),
                        "--object-mesh", str(paths["mesh"]),
                        "--gotrack-records", str(records),
                        "--output", str(episode_sheet),
                        "--max-cameras", str(manifest["debug_sheet_max_cameras"]),
                    ], log)
                task["debug_sheet_rel"] = str(episode_sheet.relative_to(shared))
                task["debug_sheet_central_rel"] = str(central.relative_to(shared))
                task["debug_sheet_publish"] = legacy._publish_debug_sheet(episode_sheet, central)
            else:
                raise ValueError(phase)
        state.update({"status": "completed", "reason": None})
    except Exception as exc:
        state.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    state.update({"finished_utc": legacy._now(), "updated_utc": legacy._now()})
    task["active_phase"] = None
    task["status"] = (
        "completed" if all(task["phases"][name]["status"] == "completed" for name in PHASES)
        else "failed" if state["status"] == "failed" else "pending"
    )
    task["updated_utc"] = legacy._now()
    _write(_task_path(schedule, task["task_id"]), task)
    return task


def _worker(args: argparse.Namespace, shared: Path, schedule: Path) -> int:
    manifest = _read(schedule / "manifest.json")
    worker_id = args.worker_id or socket.gethostname()
    worker_file = schedule / "workers" / args.phase / f"{legacy._safe_id(worker_id)}.json"
    _write(worker_file, {
        "worker_id": worker_id, "phase": args.phase, "status": "running",
        "worker_rank": args.worker_rank, "worker_count": args.worker_count,
        "started_utc": legacy._now(),
    })
    sam_segmentor = None
    foundpose_assignments = None
    if args.phase == "mask":
        from autodex.perception import Sam3ImageSegmentor
        print(f"[persistent] loading SAM3 image model once on {worker_id}", flush=True)
        sam_segmentor = Sam3ImageSegmentor(gpu=0)
    elif args.phase == "foundpose":
        foundpose_assignments = _load_foundpose_assignment(schedule, args.worker_count)
    completed = failed = 0
    while True:
        task = _claim(
            schedule, args.phase, worker_id, args.retry_failed,
            int(manifest["max_attempts"]), args.worker_rank, args.worker_count,
            foundpose_assignments,
        )
        if task is None:
            break
        result = _run_phase_task(
            args.phase, task, shared, schedule, manifest, sam_segmentor,
        )
        if result["phases"][args.phase]["status"] == "completed":
            completed += 1
        else:
            failed += 1
    _write(worker_file, {
        "worker_id": worker_id, "phase": args.phase, "status": "stopped",
        "worker_rank": args.worker_rank, "worker_count": args.worker_count,
        "completed": completed, "failed": failed, "finished_utc": legacy._now(),
    })
    print(f"[worker] phase={args.phase} {worker_id}: completed={completed} failed={failed}")
    return 0 if failed == 0 else 2


def _init(args: argparse.Namespace, shared: Path, schedule: Path) -> int:
    if schedule.exists():
        raise FileExistsError(schedule)
    object_prompts = legacy._load_prompt_map(args.object_prompts_json)
    original_prompts = legacy._load_prompt_map(args.object_prompts_original_json)
    object_aliases = legacy._load_object_aliases(args.object_alias_json)
    object_episode_map = legacy._load_object_episode_map(args.object_episodes_json)
    discovery = SimpleNamespace(
        target_root_rel=args.target_root_rel,
        cache_root_rel=args.cache_root_rel,
        mesh_root_rel=args.mesh_root_rel,
        fallback_mesh_root_rel=args.fallback_mesh_root_rel,
        objects=args.objects,
        episodes=args.episodes,
        object_episode_map=object_episode_map,
        static_images=False,
        protected_root_rel=args.protected_root_rel,
        object_aliases=object_aliases,
        robot_label=args.robot_label,
        init_frame_map={},
        init_frame_index=args.init_frame_index,
        object_prompts=object_prompts,
        object_prompts_original=original_prompts,
    )
    tasks, skipped = legacy._discover(shared, discovery)
    if args.dry_run:
        print(f"[dry-run] eligible={len(tasks)} skipped={len(skipped)}")
        for item in skipped:
            print(f"  [skip] {item}")
        return 0
    for path in (schedule / "tasks", schedule / "claims", schedule / "workers", schedule / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "layout": "phased-persistent-inline-undistort-v2",
        "schedule_id": args.schedule_id,
        "runs_root_rel": args.runs_root_rel,
        "target_root_rel": args.target_root_rel,
        "phases": list(PHASES),
        "num_cameras": args.num_cameras,
        "camera_micro_batch_size": args.camera_micro_batch_size,
        "max_video_duration_skew_sec": args.max_video_duration_skew_sec,
        "max_frames": args.max_frames,
        "init_frame_index": args.init_frame_index,
        "reverse_stop_frame_index": args.reverse_stop_frame_index,
        "min_sam3_mask_views": args.min_sam3_mask_views,
        "foundpose_selection_mode": args.foundpose_selection_mode,
        "foundpose_candidate_rank": args.foundpose_candidate_rank,
        "foundpose_per_view_candidates": args.foundpose_per_view_candidates,
        "foundpose_global_rotation_count": args.foundpose_global_rotation_count,
        "foundpose_global_coarse_max_side": args.foundpose_global_coarse_max_side,
        "foundpose_global_refine_top_k": args.foundpose_global_refine_top_k,
        "foundpose_global_min_rotation_separation_deg": args.foundpose_global_min_rotation_separation_deg,
        "foundpose_global_asymmetry_refine_top_k": args.foundpose_global_asymmetry_refine_top_k,
        "foundpose_global_asymmetry_max_side": args.foundpose_global_asymmetry_max_side,
        "foundpose_global_asymmetry_score_margin": args.foundpose_global_asymmetry_score_margin,
        "foundpose_global_asymmetry_weight": args.foundpose_global_asymmetry_weight,
        "foundpose_global_asymmetry_force": args.foundpose_global_asymmetry_force,
        "foundpose_global_dino_rerank": args.foundpose_global_dino_rerank,
        "foundpose_global_dino_score_margin": args.foundpose_global_dino_score_margin,
        "foundpose_global_dino_inlier_threshold_px": args.foundpose_global_dino_inlier_threshold_px,
        "min_valid_pose_coverage": args.min_valid_pose_coverage,
        "max_trailing_missing_frames": args.max_trailing_missing_frames,
        "debug_sheet_max_cameras": args.debug_sheet_max_cameras,
        "debug_sheet_output_root_rel": args.debug_sheet_output_root_rel,
        "max_attempts": args.max_attempts,
        "sam3_env": args.sam3_env,
        "tail_recovery": args.tail_recovery,
        "tail_recovery_frame_step": args.tail_recovery_frame_step,
        "tail_recovery_max_seed_attempts": args.tail_recovery_max_seed_attempts,
        "tail_recovery_min_mask_views": args.tail_recovery_min_mask_views,
        "tail_recovery_overlap_frames": args.tail_recovery_overlap_frames,
        "tail_recovery_min_connection_frames": args.tail_recovery_min_connection_frames,
        "tail_recovery_min_suffix_coverage": args.tail_recovery_min_suffix_coverage,
        "tail_recovery_max_translation_error_m": args.tail_recovery_max_translation_error_m,
        "tail_recovery_max_rotation_error_deg": args.tail_recovery_max_rotation_error_deg,
        "tail_recovery_max_rotation_alignment_dispersion_deg": args.tail_recovery_max_rotation_alignment_dispersion_deg,
        "n_tasks": len(tasks),
        "skipped": skipped,
        "created_utc": legacy._now(),
    }
    _write(schedule / "manifest.json", manifest)
    for task in tasks:
        task.update({
            "status": "pending", "active_phase": None,
            "phases": {
                phase: {"status": "pending", "attempts": 0, "worker_id": None}
                for phase in PHASES
            },
            "created_utc": legacy._now(), "updated_utc": legacy._now(),
        })
        _write(_task_path(schedule, task["task_id"]), task)
    print(f"[init] phased schedule={schedule} eligible={len(tasks)} skipped={len(skipped)}")
    return 0


def _phase_counts(schedule: Path, phase: str) -> dict[str, int]:
    counts = {
        name: 0
        for name in ("blocked", "pending", "running", "completed", "failed", "skipped")
    }
    for path in (schedule / "tasks").glob("*.json"):
        task = _read(path)
        status = task["phases"][phase]["status"]
        if status == "pending" and not _phase_ready(task, phase):
            counts["blocked"] += 1
        else:
            counts[status] += 1
    return counts


def _cascade_skipped(schedule: Path, phase: str) -> int:
    """Make upstream failures terminal for this and all later phases."""
    previous = PHASES[:PHASES.index(phase)]
    if not previous:
        return 0
    skipped = 0
    for path in (schedule / "tasks").glob("*.json"):
        task = _read(path)
        state = task["phases"][phase]
        if state["status"] != "pending":
            continue
        failed_upstream = next(
            (
                name for name in previous
                if task["phases"][name]["status"] in {"failed", "skipped"}
            ),
            None,
        )
        if failed_upstream is None:
            continue
        state.update({
            "status": "skipped",
            "worker_id": None,
            "reason": f"upstream_{failed_upstream}_{task['phases'][failed_upstream]['status']}",
            "updated_utc": legacy._now(),
        })
        task.update({
            "status": "failed", "active_phase": None, "updated_utc": legacy._now(),
        })
        _write(path, task)
        skipped += 1
    return skipped


def _phase_terminal(counts: dict[str, int]) -> bool:
    return counts["blocked"] == 0 and counts["pending"] == 0 and counts["running"] == 0


def _print_status(schedule: Path) -> None:
    columns = ("blocked", "pending", "running", "completed", "failed", "skipped")
    rows = [(phase, _phase_counts(schedule, phase)) for phase in PHASES]
    phase_width = max(len("phase"), *(len(phase) for phase, _ in rows))
    widths = {
        column: max(len(column), *(len(str(counts[column])) for _, counts in rows))
        for column in columns
    }
    print(
        f"{'phase':<{phase_width}}  "
        + "  ".join(f"{column:>{widths[column]}}" for column in columns)
    )
    print(
        f"{'-' * phase_width}  "
        + "  ".join("-" * widths[column] for column in columns)
    )
    for phase, counts in rows:
        print(
            f"{phase:<{phase_width}}  "
            + "  ".join(
                f"{counts[column]:>{widths[column]}}" for column in columns
            )
        )


def _reset_running(schedule: Path, phase: str, confirmed: bool) -> int:
    if not confirmed:
        raise ValueError("--confirm-workers-stopped is required")
    reset = 0
    for path in (schedule / "tasks").glob("*.json"):
        task = _read(path)
        state = task["phases"][phase]
        if state["status"] != "running":
            continue
        shutil.rmtree(
            schedule / "claims" / phase / f"{task['task_id']}.lock",
            ignore_errors=True,
        )
        state.update({
            "status": "pending", "worker_id": None,
            "reason": "reset_after_worker_stop", "updated_utc": legacy._now(),
        })
        task.update({"status": "pending", "active_phase": None, "updated_utc": legacy._now()})
        _write(path, task)
        reset += 1
    print(f"[reset-running] phase={phase} released={reset}")
    return 0


def _launch(args: argparse.Namespace, shared: Path, schedule: Path) -> int:
    cascaded = _cascade_skipped(schedule, args.phase)
    if cascaded:
        print(f"[skip-cascade] phase={args.phase} skipped={cascaded}", flush=True)
    previous = PHASES[:PHASES.index(args.phase)]
    for prerequisite in previous:
        counts = _phase_counts(schedule, prerequisite)
        if not _phase_terminal(counts):
            raise RuntimeError(f"phase {args.phase} blocked by {prerequisite}: {counts}")
    current = _phase_counts(schedule, args.phase)
    if current["pending"] == 0 and not (args.retry_failed and current["failed"] > 0):
        print(f"[launch] phase={args.phase} has no runnable tasks: {current}", flush=True)
        return 0
    env_name = args.sam3_env if args.phase == "mask" else args.gotrack_env
    if args.phase == "foundpose":
        _prepare_foundpose_assignment(
            schedule,
            len(args.workers),
            args.retry_failed,
            int(_read(schedule / "manifest.json")["max_attempts"]),
        )
    for rank, spec in enumerate(args.workers):
        worker_id = legacy._safe_id(spec)
        common = [
            "scripts/distribute_foundpose_gotrack_phased.py",
            "--mode", "worker", "--phase", args.phase,
            "--schedule-id", schedule.name,
            "--runs-root-rel", args.runs_root_rel,
            "--shared-root-rel", args.shared_root_rel,
            "--worker-id", worker_id,
            "--worker-rank", str(rank), "--worker-count", str(len(args.workers)),
        ]
        if args.retry_failed:
            common.append("--retry-failed")
        if spec == "local":
            command = [
                str(Path.home() / "anaconda3/bin/conda"), "run",
                "--no-capture-output", "-n", env_name, "python", "-u", *common,
            ]
            print("+ " + " ".join(shlex.quote(value) for value in command))
            if not args.dry_run:
                subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True)
        else:
            rendered = " ".join(shlex.quote(value) for value in common)
            worker_log = (
                f'$HOME/{args.shared_root_rel.strip("/")}/{args.runs_root_rel.strip("/")}/'
                f'{schedule.name}/logs/worker.{args.phase}.{worker_id}.log'
            )
            remote = (
                f"set -euo pipefail; cd $HOME/{args.remote_repo_rel.strip('/')}; "
                f"nohup $HOME/anaconda3/bin/conda run --no-capture-output -n {shlex.quote(env_name)} "
                f"python -u {rendered} > {worker_log} 2>&1 &"
            )
            ssh = [
                "ssh", "-p", "77", "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={args.connect_timeout}", spec, remote,
            ]
            print("+ " + " ".join(shlex.quote(value) for value in ssh))
            if not args.dry_run:
                subprocess.run(ssh, check=True)
    return 0


def _launch_all(args: argparse.Namespace, shared: Path, schedule: Path) -> int:
    """Drain all phases sequentially, attaching to an already-running phase."""
    print(
        f"[launch-all] schedule={schedule} phases={','.join(PHASES)} "
        f"workers={args.workers}",
        flush=True,
    )
    for phase in PHASES:
        args.phase = phase
        cascaded = _cascade_skipped(schedule, phase)
        if cascaded:
            print(f"[skip-cascade] phase={phase} skipped={cascaded}", flush=True)
        launched = False
        retried_failed = False
        previous_snapshot: tuple[tuple[str, int], ...] | None = None
        last_report = 0.0
        while True:
            counts = _phase_counts(schedule, phase)
            snapshot = tuple(counts.items())
            now = time.monotonic()
            if snapshot != previous_snapshot or now - last_report >= 60.0:
                print(
                    phase + " " + " ".join(f"{key}={value}" for key, value in counts.items()),
                    flush=True,
                )
                previous_snapshot = snapshot
                last_report = now
            if _phase_terminal(counts):
                if args.retry_failed and counts["failed"] > 0 and not retried_failed:
                    retried_failed = True
                    launched = False
                else:
                    break
            should_launch = (
                counts["running"] == 0
                and not launched
                and (
                    counts["pending"] > 0
                    or (args.retry_failed and counts["failed"] > 0 and retried_failed)
                )
            )
            if should_launch:
                _launch(args, shared, schedule)
                launched = True
            time.sleep(args.poll_interval)
        print(f"[launch-all] phase={phase} terminal", flush=True)
    print("[launch-all] all phases terminal", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("init", "worker", "launch", "launch-all", "status", "reset-running"),
        required=True,
    )
    parser.add_argument("--phase", choices=PHASES, default="mask")
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument("--runs-root-rel", default="object_tracking/phased_foundpose_gotrack_runs")
    parser.add_argument("--target-root-rel")
    parser.add_argument("--protected-root-rel", action="append", default=None)
    parser.add_argument("--robot-label")
    parser.add_argument("--cache-root-rel", default="mesh_new")
    parser.add_argument("--mesh-root-rel", default="mesh_new")
    parser.add_argument("--fallback-mesh-root-rel", default="mesh_blender")
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--object-alias-json")
    parser.add_argument("--object-episodes-json")
    parser.add_argument("--workers", nargs="+")
    parser.add_argument("--worker-id")
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--remote-repo-rel", default="object_tracking")
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--gotrack-env", default="gotrack")
    parser.add_argument("--sam3-env", default="sam3")
    parser.add_argument("--object-prompts-json", default=str(Path.home() / "sam3/object_prompts.json"))
    parser.add_argument("--object-prompts-original-json", default=str(Path.home() / "sam3/object_prompts_original.json"))
    parser.add_argument("--num-cameras", type=int, default=22)
    parser.add_argument("--camera-micro-batch-size", type=int, default=0)
    parser.add_argument("--max-video-duration-skew-sec", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--init-frame-index", type=int, default=30)
    parser.add_argument("--reverse-stop-frame-index", type=int, default=0)
    parser.add_argument("--min-sam3-mask-views", type=int, default=1)
    parser.add_argument("--foundpose-selection-mode", choices=("silhouette", "consensus", "hybrid", "global"), default="silhouette")
    parser.add_argument("--foundpose-candidate-rank", type=int, default=0)
    parser.add_argument("--foundpose-per-view-candidates", type=int, default=5)
    parser.add_argument("--foundpose-global-rotation-count", type=int, default=256)
    parser.add_argument("--foundpose-global-coarse-max-side", type=int, default=160)
    parser.add_argument("--foundpose-global-refine-top-k", type=int, default=5)
    parser.add_argument("--foundpose-global-min-rotation-separation-deg", type=float, default=35.0)
    parser.add_argument("--foundpose-global-asymmetry-refine-top-k", type=int, default=12)
    parser.add_argument("--foundpose-global-asymmetry-max-side", type=int, default=512)
    parser.add_argument("--foundpose-global-asymmetry-score-margin", type=float, default=0.005)
    parser.add_argument("--foundpose-global-asymmetry-weight", type=float, default=0.7)
    parser.add_argument("--foundpose-global-asymmetry-force", action="store_true")
    parser.add_argument("--foundpose-global-dino-rerank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--foundpose-global-dino-score-margin", type=float, default=0.02)
    parser.add_argument("--foundpose-global-dino-inlier-threshold-px", type=float, default=10.0)
    parser.add_argument("--min-valid-pose-coverage", type=float, default=0.5)
    parser.add_argument("--max-trailing-missing-frames", type=int, default=30)
    parser.add_argument("--tail-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tail-recovery-frame-step", type=int, default=30)
    parser.add_argument("--tail-recovery-max-seed-attempts", type=int, default=6)
    parser.add_argument("--tail-recovery-min-mask-views", type=int, default=6)
    parser.add_argument("--tail-recovery-overlap-frames", type=int, default=30)
    parser.add_argument("--tail-recovery-min-connection-frames", type=int, default=3)
    parser.add_argument("--tail-recovery-min-suffix-coverage", type=float, default=0.9)
    parser.add_argument("--tail-recovery-max-translation-error-m", type=float, default=0.03)
    parser.add_argument("--tail-recovery-max-rotation-error-deg", type=float, default=15.0)
    parser.add_argument("--tail-recovery-max-rotation-alignment-dispersion-deg", type=float, default=5.0)
    parser.add_argument("--debug-sheet-max-cameras", type=int, default=6)
    parser.add_argument("--debug-sheet-output-root-rel", default="object_tracking/gotrack_debug_sheets")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--poll-interval", type=float, default=10.0,
        help="Seconds between launch-all state checks. Default: 10.",
    )
    parser.add_argument("--confirm-workers-stopped", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.protected_root_rel = args.protected_root_rel or ["capture/eccv2026/v0"]
    if args.worker_count < 1 or not 0 <= args.worker_rank < args.worker_count:
        raise ValueError("invalid worker rank/count")
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval must be positive")
    shared = Path.home() / args.shared_root_rel
    schedule = _schedule(shared, args)
    if args.mode == "init":
        if not args.target_root_rel:
            raise ValueError("--target-root-rel is required")
        return _init(args, shared, schedule)
    if not (schedule / "manifest.json").is_file():
        raise FileNotFoundError(schedule / "manifest.json")
    if args.mode == "worker":
        return _worker(args, shared, schedule)
    if args.mode == "launch":
        if not args.workers:
            raise ValueError("--workers is required")
        return _launch(args, shared, schedule)
    if args.mode == "launch-all":
        if not args.workers:
            raise ValueError("--workers is required")
        return _launch_all(args, shared, schedule)
    if args.mode == "reset-running":
        return _reset_running(schedule, args.phase, args.confirm_workers_stopped)
    _print_status(schedule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
