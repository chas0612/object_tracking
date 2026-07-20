#!/usr/bin/env python3
"""Distribute offline FoundPose-init + GoTrack episodes across capture PCs.

Tasks are discovered as ``<robot-root>/<object>/<episode>`` and claimed through
atomic directories on shared storage.  Every task explicitly
uses the existing ``inspire_dftp/<object>/foundpose_assets`` cache; it never
silently onboards a target campaign.

Typical use:

  # Create a durable queue (inspect with --dry-run first).
  python scripts/distribute_foundpose_gotrack.py --mode init \
    --target-root-rel capture/eccv2026/hand_taeyun \
    --schedule-id hand_taeyun_foundpose_gotrack_01

  # Detach one worker per PC. Workers keep claiming episodes until empty.
  python scripts/distribute_foundpose_gotrack.py --mode launch \
    --schedule-id hand_taeyun_foundpose_gotrack_01 \
    --workers capture13@192.168.0.113 capture14@192.168.0.114 capture18@192.168.0.118

  python scripts/distribute_foundpose_gotrack.py --mode status \
    --schedule-id hand_taeyun_foundpose_gotrack_01

The controller and workers require a common ``~/shared_data`` mount.  Remote
SSH uses port 77 and key-based BatchMode authentication.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    # Several PCs can update different task files concurrently, and a retry
    # can briefly make two workers touch the same task.  A fixed ``.tmp`` name
    # lets one writer rename another writer's temporary file on shared NAS.
    temp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _schedule_dir(shared_root: Path, schedule_id: str) -> Path:
    if not SAFE_NAME.fullmatch(schedule_id):
        raise ValueError("--schedule-id may contain only letters, digits, '_', '-', and '.'")
    return shared_root / "object_tracking" / "foundpose_gotrack_runs" / schedule_id


def _mesh_for(mesh_root: Path, object_name: str) -> Path | None:
    obj_dir = mesh_root / object_name
    for suffix in (".obj", ".ply", ".glb"):
        candidate = obj_dir / f"{object_name}{suffix}"
        if candidate.is_file():
            return candidate
    candidates = [p for suffix in ("*.obj", "*.ply", "*.glb") for p in obj_dir.glob(suffix)
                  if not p.stem.endswith(("_viser", "_remeshed"))]
    return candidates[0] if len(candidates) == 1 else None


def _cache_repre(cache_root: Path, object_name: str) -> Path:
    return cache_root / object_name / "foundpose_assets" / "object_repre" / "v1" / object_name / "1" / "repre.pth"


def _load_prompt_map(path_value: str | None) -> dict[str, str]:
    """Load one optional SAM3 object-prompt map without trusting its shape."""
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Object prompt map must be a JSON object: {path}")
    return {str(key): value.strip() for key, value in payload.items()
            if isinstance(value, str) and value.strip()}


def _load_object_episode_map(path_value: str | None) -> dict[str, set[str]]:
    """Load exact object-to-episode selections without cross-product surprises."""
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Object episode map is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Object episode map must be a JSON object: {path}")
    result: dict[str, set[str]] = {}
    for object_name, episodes in payload.items():
        if not isinstance(object_name, str) or not isinstance(episodes, list):
            raise ValueError("Object episode map values must be episode lists")
        selected = {str(episode) for episode in episodes}
        if not selected:
            raise ValueError(f"Object episode map has no episodes for {object_name!r}")
        result[object_name] = selected
    return result


def _prompt_candidates(object_name: str, args: argparse.Namespace) -> list[str]:
    """Prefer curated SAM3 text, then broad original text, then object name."""
    candidates = [
        args.object_prompts.get(object_name),
        args.object_prompts_original.get(object_name),
        object_name.replace("_", " "),
    ]
    return list(dict.fromkeys(prompt for prompt in candidates if prompt))


def _discover(shared_root: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    target_root = shared_root / args.target_root_rel
    cache_root = shared_root / args.cache_root_rel
    mesh_root = shared_root / args.mesh_root_rel
    if not target_root.is_dir():
        raise FileNotFoundError(f"Target root is missing: {target_root}")
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache campaign root is missing: {cache_root}")
    objects = set(args.objects or [])
    episodes = set(args.episodes or [])
    object_episode_map = args.object_episode_map
    tasks: list[dict[str, Any]] = []
    skipped: list[str] = []
    # A robot root has object directories directly beneath it.  Deliberately
    # avoid rglob/os.walk: shared campaign roots can contain terabytes of video
    # and prior outputs, and discovery must not traverse those trees.
    episode_dirs: list[Path] = []
    generated_dirs = {
        "foundpose_assets", "undistorted_video", "object_tracking_foundpose_gotrack",
        "gotrack_tracking", "gotrack_output",
    }
    for object_dir in sorted(target_root.iterdir()):
        if not object_dir.is_dir() or object_dir.name.startswith("."):
            continue
        if objects and object_dir.name not in objects:
            continue
        if object_episode_map and object_dir.name not in object_episode_map:
            continue
        for episode_dir in sorted(object_dir.iterdir()):
            if (episode_dir.is_dir() and not episode_dir.name.startswith(".")
                    and episode_dir.name not in generated_dirs):
                episode_dirs.append(episode_dir)
    for episode_dir in episode_dirs:
        try:
            rel = episode_dir.relative_to(target_root)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        object_name, episode = rel.parts[-2:]
        robot_path = args.robot_label or args.target_root_rel
        if episodes and episode not in episodes:
            continue
        if object_episode_map and episode not in object_episode_map[object_name]:
            continue
        if not (episode_dir / "cam_param" / "extrinsics.json").is_file() or not (episode_dir / "videos").is_dir():
            skipped.append(f"{rel}: missing videos/ or cam_param/extrinsics.json")
            continue
        mesh = _mesh_for(mesh_root, object_name)
        if mesh is None:
            skipped.append(f"{rel}: no unambiguous mesh under {mesh_root / object_name}")
            continue
        repre = _cache_repre(cache_root, object_name)
        if not repre.is_file():
            skipped.append(f"{rel}: inspire cache missing ({repre})")
            continue
        task_id = _safe_id(rel.as_posix())
        tasks.append({
            "task_id": task_id, "robot": robot_path, "object_name": object_name, "episode": episode,
            "episode_rel": str(episode_dir.relative_to(shared_root)),
            "mesh_rel": str(mesh.relative_to(shared_root)),
            "assets_rel": str(repre.parents[4].relative_to(shared_root)),
            "cache_repre_rel": str(repre.relative_to(shared_root)),
            "sam3_prompts": _prompt_candidates(object_name, args),
        })
    return tasks, skipped


def _tracking_summary(task: dict[str, Any], attempt_dir: Path, min_coverage: float, max_trailing_missing: int) -> dict[str, Any]:
    records = attempt_dir / "gotrack_tracking" / "gotrack_output" / task["object_name"] / "world_pose_records.json"
    if not records.is_file():
        return {"complete": False, "reason": "records_missing", "records": 0, "valid_poses": 0,
                "coverage": 0.0, "trailing_missing": 0}
    try:
        data = json.loads(records.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"complete": False, "reason": "records_invalid_json", "records": 0, "valid_poses": 0,
                "coverage": 0.0, "trailing_missing": 0}
    if not isinstance(data, list) or not data:
        return {"complete": False, "reason": "records_empty", "records": 0, "valid_poses": 0,
                "coverage": 0.0, "trailing_missing": 0}
    valid = sum(isinstance(row, dict) and row.get("pose_world") is not None for row in data)
    trailing_missing = 0
    for row in reversed(data):
        if isinstance(row, dict) and row.get("pose_world") is not None:
            break
        trailing_missing += 1
    coverage = valid / len(data)
    complete = coverage >= min_coverage and trailing_missing <= max_trailing_missing
    if coverage < min_coverage:
        reason = f"coverage={coverage:.3f}<{min_coverage:.3f}"
    elif trailing_missing > max_trailing_missing:
        reason = f"trailing_missing={trailing_missing}>{max_trailing_missing}"
    else:
        reason = None
    return {"complete": complete, "reason": reason, "records": len(data), "valid_poses": valid,
            "coverage": coverage, "trailing_missing": trailing_missing}


def _task_path(schedule_dir: Path, task_id: str) -> Path:
    return schedule_dir / "tasks" / f"{task_id}.json"


def _claim_task(schedule_dir: Path, worker_id: str, retry_failed: bool, max_attempts: int) -> dict[str, Any] | None:
    for path in sorted((schedule_dir / "tasks").glob("*.json")):
        task = _read_json(path)
        status = task.get("status", "pending")
        if status not in ({"pending"} | ({"failed"} if retry_failed else set())):
            continue
        # ``pending`` also represents a task recovered after an intentional
        # worker stop.  It must be claimable even when the interrupted attempt
        # already reached the normal failed-attempt ceiling.
        if status == "failed" and int(task.get("attempts", 0)) >= max_attempts:
            continue
        claim = schedule_dir / "claims" / f"{task['task_id']}.lock"
        # A failed task deliberately retains its claim as an audit record.  A
        # user-requested retry is the only path that releases that stale claim.
        if status == "failed" and retry_failed and claim.exists():
            try:
                shutil.rmtree(claim)
            except FileNotFoundError:
                continue
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        _atomic_json(claim / "claim.json", {"worker_id": worker_id, "host": socket.gethostname(), "claimed_utc": _now()})
        task.update({"status": "running", "worker_id": worker_id, "attempts": int(task.get("attempts", 0)) + 1,
                     "started_utc": _now(), "updated_utc": _now()})
        _atomic_json(path, task)
        return task
    return None


def _run_command(command: list[str], log: Any, cwd: Path) -> None:
    log.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
    log.flush()
    subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True, check=True)


def _publish_debug_sheet(episode_sheet: Path, central_sheet: Path) -> str:
    """Make the central QA view point at the episode-owned sheet.

    A relative symlink avoids duplicating JPEGs on the shared NAS.  Some NAS
    mounts disallow symlinks, in which case a normal copy is still useful and
    keeps sheet generation non-fatal to tracking.
    """
    central_sheet.parent.mkdir(parents=True, exist_ok=True)
    if central_sheet.exists() or central_sheet.is_symlink():
        return "existing"
    try:
        central_sheet.symlink_to(os.path.relpath(episode_sheet, central_sheet.parent))
        return "symlink"
    except OSError:
        shutil.copy2(episode_sheet, central_sheet)
        return "copy"


def _run_task(schedule_dir: Path, task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    shared = Path.home() / args.shared_root_rel
    episode = shared / task["episode_rel"]
    mesh = shared / task["mesh_rel"]
    assets = shared / task["assets_rel"]
    attempt = int(task["attempts"])
    attempt_dir = episode / "object_tracking_foundpose_gotrack" / schedule_dir.name / f"attempt_{attempt:02d}"
    init_frame_index = int(args.init_frame_index)
    frame_dir, init_dir, track_dir = (
        attempt_dir / f"foundpose_frame_{init_frame_index:06d}",
        attempt_dir / "foundpose_init",
        attempt_dir / "gotrack_tracking",
    )
    log_path = schedule_dir / "logs" / f"{task['task_id']}.attempt{attempt}.{task['worker_id']}.log"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    task.update({"attempt_dir": str(attempt_dir.relative_to(shared)), "log": str(log_path), "phase": "undistort"})
    _atomic_json(_task_path(schedule_dir, task["task_id"]), task)
    gotrack = [str(Path.home() / "anaconda3/bin/conda"), "run", "--no-capture-output", "-n", args.gotrack_env, "python", "-u"]
    sam3 = [str(Path.home() / "anaconda3/bin/conda"), "run", "--no-capture-output", "-n", args.sam3_env, "python", "-u"]
    # A locally launched worker may use whichever conda is on the controller PATH.
    if args.local_worker:
        gotrack[0] = "conda"
        sam3[0] = "conda"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"task={task['task_id']} worker={task['worker_id']} started_utc={_now()}\n")
            _run_command(gotrack + ["src/process/undistort_capture_videos.py", "--capture-dir", str(episode)], log, REPO_ROOT)
            task["phase"] = "mask"
            _atomic_json(_task_path(schedule_dir, task["task_id"]), task)
            prompts = task.get("sam3_prompts") or _prompt_candidates(task["object_name"], args)
            task["sam3_attempted_prompts"] = []
            masks_available = False
            for prompt in prompts:
                task["sam3_prompt_current"] = prompt
                task["sam3_attempted_prompts"].append(prompt)
                _atomic_json(_task_path(schedule_dir, task["task_id"]), task)
                _run_command(sam3 + ["src/process/mask.py", "--capture_dir", str(episode), "--frame-index", str(init_frame_index),
                                     "--prompt", prompt, "--video-dir", str(episode / "undistorted_video"),
                                     "--frame-output-dir", str(frame_dir)], log, REPO_ROOT)
                metadata_path = frame_dir / "metadata.json"
                metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
                masks_available = int(metadata.get("masks_written", 0)) + int(metadata.get("masks_skipped", 0)) > 0
                if masks_available:
                    task["sam3_prompt_used"] = prompt
                    break
                log.write(f"[sam3] no masks with prompt={prompt!r}; trying fallback\n")
                log.flush()
            if not masks_available:
                raise RuntimeError(f"SAM3 produced no masks with prompts: {prompts}")
            task["phase"] = "foundpose"
            _atomic_json(_task_path(schedule_dir, task["task_id"]), task)
            _run_command(gotrack + ["src/process/foundpose_init_capture.py", "--capture-dir", str(episode), "--frame-dir", str(frame_dir),
                                    "--mesh", str(mesh), "--object-name", task["object_name"], "--assets-root", str(assets),
                                    "--output-dir", str(init_dir)], log, REPO_ROOT)
            task["phase"] = "gotrack"
            _atomic_json(_task_path(schedule_dir, task["task_id"]), task)
            _run_command(gotrack + ["src/process/gotrack_capture.py", "--capture-dir", str(episode), "--video-dir", str(episode / "undistorted_video"),
                                    "--mesh", str(mesh), "--init-pose", str(init_dir / "init_pose_world.npy"), "--object-name", task["object_name"],
                                    "--num-cameras", str(args.num_cameras), "--allow-fewer-cameras",
                                    "--camera-micro-batch-size", str(args.camera_micro_batch_size),
                                    "--init-frame-index", str(init_frame_index),
                                    "--max-video-duration-skew-sec", str(args.max_video_duration_skew_sec),
                                    "--max-frames", str(args.max_frames), "--output-dir", str(track_dir)], log, REPO_ROOT)
            tracking_summary = _tracking_summary(task, attempt_dir, args.min_valid_pose_coverage, args.max_trailing_missing_frames)
            task["tracking_summary"] = tracking_summary
            tracking_ok = bool(tracking_summary["complete"])
            if tracking_ok and args.debug_sheets:
                task["phase"] = "debug_sheet"
                _atomic_json(_task_path(schedule_dir, task["task_id"]), task)
                records = track_dir / "gotrack_output" / task["object_name"] / "world_pose_records.json"
                episode_sheet = episode / f"gotrack_debug_sheet_{schedule_dir.name}_{attempt_dir.name}.jpg"
                central_sheet = shared / args.debug_sheet_output_root_rel / schedule_dir.name / f"{task['task_id']}.jpg"
                try:
                    if not episode_sheet.is_file():
                        _run_command(gotrack + ["src/process/render_gotrack_debug_sheet.py", "--capture-dir", str(episode),
                                                "--object-mesh", str(mesh), "--gotrack-records", str(records),
                                                "--output", str(episode_sheet), "--max-cameras", str(args.debug_sheet_max_cameras)], log, REPO_ROOT)
                    task["debug_sheet_rel"] = str(episode_sheet.relative_to(shared))
                    task["debug_sheet_central_rel"] = str(central_sheet.relative_to(shared))
                    task["debug_sheet_publish"] = _publish_debug_sheet(episode_sheet, central_sheet)
                    task["debug_sheet_status"] = "completed"
                except Exception as exc:
                    # A renderer problem must not discard an otherwise valid
                    # tracking result.  The retryable batch renderer remains
                    # available for this exact recovery path.
                    task["debug_sheet_status"] = "failed"
                    task["debug_sheet_error"] = f"{type(exc).__name__}: {exc}"
                    log.write(f"[debug-sheet] nonfatal failure: {task['debug_sheet_error']}\n")
                    log.flush()
        tracking_summary = _tracking_summary(task, attempt_dir, args.min_valid_pose_coverage, args.max_trailing_missing_frames)
        task["tracking_summary"] = tracking_summary
        task["status"] = "completed" if tracking_summary["complete"] else "failed"
        task["reason"] = None if task["status"] == "completed" else f"tracking_incomplete: {tracking_summary['reason']}"
    except subprocess.CalledProcessError as exc:
        task.update({"status": "failed", "reason": f"returncode={exc.returncode}"})
    except Exception as exc:
        task.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        task.update({"phase": "complete" if task.get("status") == "completed" else "failed", "finished_utc": _now(), "updated_utc": _now()})
        _atomic_json(_task_path(schedule_dir, task["task_id"]), task)
    return task


def _init(args: argparse.Namespace, shared: Path, schedule: Path) -> int:
    tasks, skipped = _discover(shared, args)
    if args.dry_run:
        print(f"[dry-run] eligible={len(tasks)} skipped={len(skipped)}")
        for task in tasks[:30]: print(f"  {task['robot']}/{task['object_name']}/{task['episode']}")
        for line in skipped[:20]: print(f"  [skip] {line}")
        return 0
    if schedule.exists(): raise FileExistsError(f"Schedule exists: {schedule}")
    for dirname in ("tasks", "claims", "logs", "workers"): (schedule / dirname).mkdir(parents=True, exist_ok=True)
    _atomic_json(schedule / "manifest.json", {"schedule_id": schedule.name, "created_utc": _now(), "target_root_rel": args.target_root_rel,
                                                "cache_root_rel": args.cache_root_rel, "mesh_root_rel": args.mesh_root_rel,
                                                "object_episode_map": {name: sorted(episodes) for name, episodes in args.object_episode_map.items()},
                                                "num_cameras": args.num_cameras, "max_frames": args.max_frames,
                                                "camera_micro_batch_size": args.camera_micro_batch_size,
                                                "max_video_duration_skew_sec": args.max_video_duration_skew_sec,
                                                "init_frame_index": args.init_frame_index,
                                                "debug_sheets": args.debug_sheets,
                                                "debug_sheet_max_cameras": args.debug_sheet_max_cameras,
                                                "debug_sheet_output_root_rel": args.debug_sheet_output_root_rel,
                                                "min_valid_pose_coverage": args.min_valid_pose_coverage,
                                                "max_trailing_missing_frames": args.max_trailing_missing_frames,
                                                "max_attempts": args.max_attempts, "n_tasks": len(tasks), "skipped": skipped})
    for task in tasks:
        task.update({"status": "pending", "attempts": 0, "phase": "pending", "created_utc": _now(), "updated_utc": _now()})
        _atomic_json(_task_path(schedule, task["task_id"]), task)
    print(f"[init] schedule={schedule} eligible={len(tasks)} skipped={len(skipped)}")
    return 0


def _worker(args: argparse.Namespace, shared: Path, schedule: Path) -> int:
    worker_id = args.worker_id or socket.gethostname()
    _atomic_json(schedule / "workers" / f"{_safe_id(worker_id)}.json", {"worker_id": worker_id, "status": "running", "started_utc": _now()})
    done = failed = 0
    while True:
        task = _claim_task(schedule, worker_id, args.retry_failed, args.max_attempts)
        if task is None: break
        result = _run_task(schedule, task, args)
        done += result["status"] == "completed"; failed += result["status"] != "completed"
    _atomic_json(schedule / "workers" / f"{_safe_id(worker_id)}.json", {"worker_id": worker_id, "status": "stopped", "finished_utc": _now(), "completed": done, "failed": failed})
    print(f"[worker] {worker_id}: completed={done} failed={failed}")
    return 0 if failed == 0 else 2


def _launch(args: argparse.Namespace, schedule: Path) -> int:
    manifest = _read_json(schedule / "manifest.json")
    # The queue's processing knobs belong to the immutable init manifest, not
    # to whichever controller later launches/relaunches the workers.
    args.num_cameras = int(manifest["num_cameras"])
    args.max_frames = int(manifest["max_frames"])
    args.camera_micro_batch_size = int(manifest["camera_micro_batch_size"])
    args.max_video_duration_skew_sec = float(manifest["max_video_duration_skew_sec"])
    args.init_frame_index = int(manifest.get("init_frame_index", 0))
    args.debug_sheets = bool(manifest.get("debug_sheets", args.debug_sheets))
    args.debug_sheet_max_cameras = int(manifest.get("debug_sheet_max_cameras", args.debug_sheet_max_cameras))
    args.debug_sheet_output_root_rel = str(manifest.get("debug_sheet_output_root_rel", args.debug_sheet_output_root_rel))
    args.min_valid_pose_coverage = float(manifest.get("min_valid_pose_coverage", args.min_valid_pose_coverage))
    args.max_trailing_missing_frames = int(manifest.get("max_trailing_missing_frames", args.max_trailing_missing_frames))
    args.max_attempts = int(manifest.get("max_attempts", args.max_attempts))
    for spec in args.workers:
        worker_id = _safe_id(spec)
        command = ["$HOME/anaconda3/bin/conda", "run", "--no-capture-output", "-n", args.gotrack_env, "python", "-u",
                   "scripts/distribute_foundpose_gotrack.py", "--mode", "worker", "--schedule-id", schedule.name,
                   "--worker-id", worker_id, "--gotrack-env", args.gotrack_env, "--sam3-env", args.sam3_env,
                   "--shared-root-rel", args.shared_root_rel, "--num-cameras", str(args.num_cameras),
                   "--camera-micro-batch-size", str(args.camera_micro_batch_size),
                   "--max-video-duration-skew-sec", str(args.max_video_duration_skew_sec),
                   "--init-frame-index", str(args.init_frame_index),
                   "--debug-sheet-max-cameras", str(args.debug_sheet_max_cameras),
                   "--debug-sheet-output-root-rel", args.debug_sheet_output_root_rel,
                   "--min-valid-pose-coverage", str(args.min_valid_pose_coverage),
                   "--max-trailing-missing-frames", str(args.max_trailing_missing_frames),
                   "--max-frames", str(args.max_frames), "--max-attempts", str(args.max_attempts)]
        command.append("--debug-sheets" if args.debug_sheets else "--no-debug-sheets")
        if args.retry_failed:
            command.append("--retry-failed")
        rendered = " ".join(part if part.startswith("$HOME/") else shlex.quote(part) for part in command)
        if spec == "local":
            local = [sys.executable, *command[7:]]
            print("+ " + " ".join(shlex.quote(x) for x in local))
            if not args.dry_run: subprocess.Popen(local, cwd=REPO_ROOT, start_new_session=True)
            continue
        remote_log = f'$HOME/{args.shared_root_rel.strip("/")}/object_tracking/foundpose_gotrack_runs/{schedule.name}/logs/worker.{worker_id}.log'
        remote = f"set -euo pipefail; cd $HOME/{args.remote_repo_rel.strip('/')}; nohup {rendered} > {remote_log} 2>&1 &"
        ssh = ["ssh", "-p", "77", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={args.connect_timeout}", spec, remote]
        print("+ " + " ".join(shlex.quote(x) for x in ssh))
        if not args.dry_run: subprocess.run(ssh, check=True)
    return 0


def _status(schedule: Path) -> int:
    counts: dict[str, int] = {}
    for path in sorted((schedule / "tasks").glob("*.json")):
        status = _read_json(path).get("status", "unknown"); counts[status] = counts.get(status, 0) + 1
    print(f"schedule={schedule}")
    print(" ".join(f"{k}={counts.get(k, 0)}" for k in ("pending", "running", "completed", "failed")))
    return 1 if counts.get("failed") else 0


def _reset_running(schedule: Path, confirm_workers_stopped: bool) -> int:
    """Release claims left by intentionally stopped detached workers."""
    if not confirm_workers_stopped:
        raise ValueError("--confirm-workers-stopped is required: resetting a live task can duplicate GPU work")
    reset = 0
    for path in sorted((schedule / "tasks").glob("*.json")):
        task = _read_json(path)
        if task.get("status") != "running":
            continue
        claim = schedule / "claims" / f"{task['task_id']}.lock"
        if claim.exists():
            shutil.rmtree(claim)
        task.update({"status": "pending", "phase": "pending", "worker_id": None,
                     "reason": "reset_after_worker_stop", "updated_utc": _now()})
        _atomic_json(path, task)
        reset += 1
    print(f"[reset-running] released={reset}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("init", "worker", "launch", "status", "reset-running"), required=True)
    p.add_argument("--schedule-id", required=True)
    p.add_argument("--shared-root-rel", default="shared_data")
    p.add_argument("--target-root-rel", default=None, help="With --mode init: robot root under ~/shared_data; its immediate children are objects.")
    p.add_argument("--robot-label", default=None, help="Label recorded in tasks; default: --target-root-rel.")
    p.add_argument("--cache-root-rel", default="capture/eccv2026/inspire_dftp", help="Source campaign cache root under ~/shared_data.")
    p.add_argument("--mesh-root-rel", default="mesh_blender")
    p.add_argument("--objects", nargs="*", default=None); p.add_argument("--episodes", nargs="*", default=None)
    p.add_argument("--object-episodes-json", default=None,
                   help="JSON object mapping each object to its exact episode list; avoids --objects/--episodes cross products.")
    p.add_argument("--workers", nargs="+", default=None); p.add_argument("--worker-id", default=None)
    p.add_argument("--remote-repo-rel", default="object_tracking"); p.add_argument("--connect-timeout", type=int, default=10)
    p.add_argument("--gotrack-env", default="gotrack"); p.add_argument("--sam3-env", default="sam3")
    p.add_argument("--object-prompts-json", default=str(Path.home() / "sam3/object_prompts.json"),
                   help="Curated object-to-SAM3-prompt JSON. Missing file is allowed.")
    p.add_argument("--object-prompts-original-json", default=str(Path.home() / "sam3/object_prompts_original.json"),
                   help="Broader fallback object-to-SAM3-prompt JSON. Missing file is allowed.")
    p.add_argument("--num-cameras", type=int, default=22)
    p.add_argument("--camera-micro-batch-size", type=int, default=0,
                   help="0 (default) refines all selected cameras together; set lower only to reduce GPU memory.")
    p.add_argument("--max-video-duration-skew-sec", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--init-frame-index", type=int, default=30,
                   help="SAM3/FoundPose bootstrap frame. Default 30 avoids stale frame-0 captures; GoTrack merges reverse prefix poses automatically.")
    p.add_argument("--max-attempts", type=int, default=2)
    debug_group = p.add_mutually_exclusive_group()
    debug_group.add_argument("--debug-sheets", dest="debug_sheets", action="store_true",
                             help="Render one compact reprojection sheet after each successful GoTrack task (default).")
    debug_group.add_argument("--no-debug-sheets", dest="debug_sheets", action="store_false",
                             help="Skip per-task sheet rendering; use the batch renderer later if needed.")
    p.set_defaults(debug_sheets=True)
    p.add_argument("--debug-sheet-max-cameras", type=int, default=6)
    p.add_argument("--debug-sheet-output-root-rel", default="object_tracking/gotrack_debug_sheets")
    p.add_argument("--min-valid-pose-coverage", type=float, default=0.5,
                   help="Mark task failed below this valid-pose fraction. Deliberately conservative; semantic drift stays completed/suspect.")
    p.add_argument("--max-trailing-missing-frames", type=int, default=30,
                   help="Mark task failed when more than this many final frames lack poses.")
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-workers-stopped", action="store_true",
                   help="Required with --mode reset-running; confirms no worker can still own a task.")
    args = p.parse_args()
    if args.connect_timeout < 1 or args.num_cameras < 1 or args.camera_micro_batch_size < 0 or args.max_video_duration_skew_sec < 0 or args.max_frames == 0 or args.max_attempts < 1 or args.init_frame_index < 0 or args.debug_sheet_max_cameras < 1 or not 0 < args.min_valid_pose_coverage <= 1 or args.max_trailing_missing_frames < 0: raise ValueError("invalid worker/tracking option")
    roots = (args.cache_root_rel, args.mesh_root_rel, args.debug_sheet_output_root_rel) + ((args.target_root_rel,) if args.target_root_rel else ())
    if any(Path(x).is_absolute() or ".." in Path(x).parts for x in roots): raise ValueError("root paths must be safe relative paths")
    shared = Path.home() / args.shared_root_rel; schedule = _schedule_dir(shared, args.schedule_id)
    args.object_prompts = _load_prompt_map(args.object_prompts_json)
    args.object_prompts_original = _load_prompt_map(args.object_prompts_original_json)
    args.object_episode_map = _load_object_episode_map(args.object_episodes_json)
    args.local_worker = args.mode == "worker" and (args.worker_id or socket.gethostname()).startswith("local")
    if args.mode == "init":
        if not args.target_root_rel: raise ValueError("--target-root-rel is required for --mode init")
        return _init(args, shared, schedule)
    if not (schedule / "manifest.json").is_file(): raise FileNotFoundError(f"Schedule manifest missing: {schedule}")
    if args.mode == "worker": return _worker(args, shared, schedule)
    if args.mode == "launch":
        if not args.workers: raise ValueError("--workers is required for --mode launch")
        return _launch(args, schedule)
    if args.mode == "reset-running":
        return _reset_running(schedule, args.confirm_workers_stopped)
    return _status(schedule)


if __name__ == "__main__":
    raise SystemExit(main())
