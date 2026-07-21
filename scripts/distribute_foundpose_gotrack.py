#!/usr/bin/env python3
"""Distribute offline FoundPose-init + GoTrack episodes across capture PCs.

Tasks are discovered as ``<robot-root>/<object>/<episode>`` and claimed through
atomic directories on shared storage.  Every task explicitly
uses the canonical ``mesh_new/<object>/foundpose_assets`` cache; it never
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: Any) -> None:
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


def _mesh_for_roots(mesh_roots: list[Path], object_name: str) -> Path | None:
    """Resolve an object mesh from ordered evidence roots.

    ``mesh_new`` contains corrected meshes for a growing subset of objects;
    retain ``mesh_blender`` as a compatibility fallback for the rest.
    """
    for mesh_root in mesh_roots:
        mesh = _mesh_for(mesh_root, object_name)
        if mesh is not None:
            return mesh
    return None


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
    mesh_roots = [shared_root / args.mesh_root_rel]
    fallback_mesh_root = shared_root / args.fallback_mesh_root_rel
    if fallback_mesh_root not in mesh_roots:
        mesh_roots.append(fallback_mesh_root)
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
        mesh = _mesh_for_roots(mesh_roots, object_name)
        if mesh is None:
            searched = ", ".join(str(root / object_name) for root in mesh_roots)
            skipped.append(f"{rel}: no unambiguous mesh under {searched}")
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


def _record_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a JSON record list: {path}")
    return payload


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


def _foundpose_command(
    gotrack: list[str], *, capture: Path, frame_dir: Path, mesh: Path,
    object_name: str, assets: Path, output_dir: Path, args: argparse.Namespace,
) -> list[str]:
    return gotrack + [
        "src/process/foundpose_init_capture.py",
        "--capture-dir", str(capture), "--frame-dir", str(frame_dir),
        "--mesh", str(mesh), "--object-name", object_name, "--assets-root", str(assets),
        "--pose-selection-mode", args.foundpose_selection_mode,
        "--pose-candidate-rank", str(args.foundpose_candidate_rank),
        "--per-view-candidates", str(args.foundpose_per_view_candidates),
        "--global-rotation-count", str(args.foundpose_global_rotation_count),
        "--global-coarse-max-side", str(args.foundpose_global_coarse_max_side),
        "--global-refine-top-k", str(args.foundpose_global_refine_top_k),
        "--global-min-rotation-separation-deg", str(args.foundpose_global_min_rotation_separation_deg),
        "--global-asymmetry-refine-top-k", str(args.foundpose_global_asymmetry_refine_top_k),
        "--global-asymmetry-max-side", str(args.foundpose_global_asymmetry_max_side),
        "--global-asymmetry-score-margin", str(args.foundpose_global_asymmetry_score_margin),
        "--global-asymmetry-weight", str(args.foundpose_global_asymmetry_weight),
        "--output-dir", str(output_dir),
    ]


def _attempt_tail_recovery(
    *, schedule_dir: Path, task: dict[str, Any], attempt_dir: Path, episode: Path,
    mesh: Path, assets: Path, track_dir: Path, gotrack: list[str], sam3: list[str],
    prompts: list[str], args: argparse.Namespace, log: Any,
) -> bool:
    """Try late SAM3/FoundPose seeds, then run one validated reverse bridge."""
    from autodex.tracking.tail_recovery import (
        assess_tail_recovery,
        descending_seed_frames,
        merge_tail_recovery,
        tail_gap,
    )

    records_path = track_dir / "gotrack_output" / task["object_name"] / "world_pose_records.json"
    if not records_path.is_file():
        return False
    original = _record_list(records_path)
    gap = tail_gap(original)
    if gap["trailing_missing"] <= args.max_trailing_missing_frames:
        return False

    primary_manifest_path = track_dir / "run_manifest.json"
    primary_manifest = _read_json(primary_manifest_path) if primary_manifest_path.is_file() else {}
    selected = [str(value) for value in primary_manifest.get("selected_cameras", [])]
    timings = primary_manifest.get("video_timings", {})
    common_last = min(
        (int(timings[camera]["frames"]) - 1 for camera in selected
         if camera in timings and int(timings[camera].get("frames", 0)) > 0),
        default=gap["last_frame"],
    )
    seeds = descending_seed_frames(
        original, step=args.tail_recovery_frame_step,
        max_attempts=args.tail_recovery_max_seed_attempts,
        maximum_frame=common_last,
    )
    recovery_root = attempt_dir / "tail_recovery"
    recovery_root.mkdir(parents=True, exist_ok=True)
    recovery_manifest: dict[str, Any] = {
        "status": "searching", "created_utc": _now(), "original_gap": gap,
        "candidate_seed_frames": seeds, "selected_cameras": selected,
        "seed_attempts": [],
    }
    manifest_path = recovery_root / "recovery_manifest.json"
    _atomic_json(manifest_path, recovery_manifest)
    if not seeds:
        recovery_manifest.update({"status": "failed", "reason": "no_common_late_seed_frame"})
        _atomic_json(manifest_path, recovery_manifest)
        return False

    chosen: tuple[int, Path, str] | None = None
    for seed_frame in seeds:
        seed_info: dict[str, Any] = {"frame_index": seed_frame, "prompt_attempts": []}
        recovery_manifest["seed_attempts"].append(seed_info)
        seed_root = recovery_root / f"seed_{seed_frame:06d}"
        seed_root.mkdir(parents=True, exist_ok=True)
        for prompt_index, prompt in enumerate(prompts, start=1):
            prompt_root = seed_root / f"prompt_{prompt_index:02d}"
            frame_dir = prompt_root / f"foundpose_frame_{seed_frame:06d}"
            init_dir = prompt_root / "foundpose_init"
            prompt_info: dict[str, Any] = {"prompt": prompt, "frame_dir": str(frame_dir)}
            seed_info["prompt_attempts"].append(prompt_info)
            task.update({"tail_recovery_seed_current": seed_frame,
                         "tail_recovery_prompt_current": prompt})
            _atomic_json(_task_path(schedule_dir, task["task_id"]), task)
            mask_command = sam3 + [
                "src/process/mask.py", "--capture_dir", str(episode),
                "--frame-index", str(seed_frame), "--prompt", prompt,
                "--video-dir", str(episode / "undistorted_video"),
                "--frame-output-dir", str(frame_dir),
            ]
            if selected:
                mask_command += ["--serials", *selected]
            try:
                _run_command(mask_command, log, REPO_ROOT)
            except subprocess.CalledProcessError as exc:
                prompt_info.update({"status": "mask_failed", "returncode": exc.returncode})
                _atomic_json(manifest_path, recovery_manifest)
                continue
            metadata_path = frame_dir / "metadata.json"
            metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
            mask_views = int(metadata.get("masks_written", 0)) + int(metadata.get("masks_skipped", 0))
            prompt_info["mask_views"] = mask_views
            if mask_views < args.tail_recovery_min_mask_views:
                prompt_info["status"] = "insufficient_masks"
                _atomic_json(manifest_path, recovery_manifest)
                continue
            try:
                _run_command(_foundpose_command(
                    gotrack, capture=episode, frame_dir=frame_dir, mesh=mesh,
                    object_name=task["object_name"], assets=assets,
                    output_dir=init_dir, args=args,
                ), log, REPO_ROOT)
            except subprocess.CalledProcessError as exc:
                prompt_info.update({"status": "foundpose_failed", "returncode": exc.returncode})
                _atomic_json(manifest_path, recovery_manifest)
                continue
            result_path = init_dir / "result.json"
            result = _read_json(result_path) if result_path.is_file() else {}
            foundpose_views = int(result.get("num_foundpose_candidates", 0))
            min_foundpose_views = max(3, args.tail_recovery_min_mask_views // 2)
            prompt_info["foundpose_views"] = foundpose_views
            if foundpose_views < min_foundpose_views:
                prompt_info.update({
                    "status": "insufficient_foundpose_views",
                    "min_foundpose_views": min_foundpose_views,
                })
                _atomic_json(manifest_path, recovery_manifest)
                continue
            prompt_info["status"] = "foundpose_succeeded"
            chosen = (seed_frame, init_dir, prompt)
            _atomic_json(manifest_path, recovery_manifest)
            break
        if chosen is not None:
            break

    if chosen is None:
        recovery_manifest.update({"status": "failed", "reason": "no_late_foundpose_seed"})
        _atomic_json(manifest_path, recovery_manifest)
        return False

    seed_frame, init_dir, prompt = chosen
    recovery_track_dir = recovery_root / f"seed_{seed_frame:06d}" / "gotrack_tracking"
    reverse_stop = max(0, gap["last_valid_frame"] - args.tail_recovery_overlap_frames + 1)
    track_command = gotrack + [
        "src/process/gotrack_capture.py", "--capture-dir", str(episode),
        "--video-dir", str(episode / "undistorted_video"), "--mesh", str(mesh),
        "--init-pose", str(init_dir / "init_pose_world.npy"),
        "--object-name", task["object_name"], "--num-cameras", str(args.num_cameras),
        "--allow-fewer-cameras", "--camera-micro-batch-size", str(args.camera_micro_batch_size),
        "--init-frame-index", str(seed_frame), "--reverse-stop-frame-index", str(reverse_stop),
        "--max-video-duration-skew-sec", str(args.max_video_duration_skew_sec),
        "--max-frames", str(args.max_frames), "--output-dir", str(recovery_track_dir),
    ]
    if selected:
        track_command += ["--camera-ids", *selected]
    recovery_manifest.update({
        "status": "tracking", "selected_seed_frame": seed_frame,
        "selected_prompt": prompt, "reverse_stop_frame": reverse_stop,
    })
    _atomic_json(manifest_path, recovery_manifest)
    try:
        _run_command(track_command, log, REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        recovery_manifest.update({"status": "failed", "reason": f"gotrack_returncode={exc.returncode}"})
        _atomic_json(manifest_path, recovery_manifest)
        return False

    recovery_records_path = (
        recovery_track_dir / "gotrack_output" / task["object_name"] / "world_pose_records.json"
    )
    recovery_records = _record_list(recovery_records_path)
    assessment = assess_tail_recovery(
        original, recovery_records,
        overlap_frames=args.tail_recovery_overlap_frames,
        min_connection_frames=args.tail_recovery_min_connection_frames,
        min_suffix_coverage=args.tail_recovery_min_suffix_coverage,
        max_trailing_missing=args.max_trailing_missing_frames,
        max_translation_error_m=args.tail_recovery_max_translation_error_m,
        max_rotation_error_deg=args.tail_recovery_max_rotation_error_deg,
        max_rotation_alignment_dispersion_deg=args.tail_recovery_max_rotation_alignment_dispersion_deg,
    )
    recovery_manifest["assessment"] = assessment
    if not assessment["accepted"]:
        recovery_manifest.update({"status": "rejected", "reason": assessment["reason"]})
        _atomic_json(manifest_path, recovery_manifest)
        return False

    backup = track_dir / "pre_tail_recovery_world_pose_records.json"
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite recovery backup: {backup}")
    shutil.copy2(records_path, backup)
    merged = merge_tail_recovery(
        original, recovery_records, recovery_seed_frame=seed_frame,
        rotation_alignment=assessment.get("rotation_alignment"),
    )
    _atomic_json(records_path, merged)
    recovery_manifest.update({
        "status": "accepted", "finished_utc": _now(),
        "original_records_backup": str(backup), "published_records": str(records_path),
    })
    _atomic_json(manifest_path, recovery_manifest)
    task["tail_recovery"] = {
        "status": "accepted", "seed_frame": seed_frame, "prompt": prompt,
        "assessment": assessment, "manifest": str(manifest_path),
    }
    return True


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
            _run_command(_foundpose_command(
                gotrack, capture=episode, frame_dir=frame_dir, mesh=mesh,
                object_name=task["object_name"], assets=assets,
                output_dir=init_dir, args=args,
            ), log, REPO_ROOT)
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
            if (not tracking_ok and args.tail_recovery
                    and int(tracking_summary.get("trailing_missing", 0)) > args.max_trailing_missing_frames):
                task["phase"] = "tail_recovery"
                _atomic_json(_task_path(schedule_dir, task["task_id"]), task)
                recovered = _attempt_tail_recovery(
                    schedule_dir=schedule_dir, task=task, attempt_dir=attempt_dir,
                    episode=episode, mesh=mesh, assets=assets, track_dir=track_dir,
                    gotrack=gotrack, sam3=sam3, prompts=prompts, args=args, log=log,
                )
                tracking_summary = _tracking_summary(
                    task, attempt_dir, args.min_valid_pose_coverage, args.max_trailing_missing_frames,
                )
                task["tracking_summary"] = tracking_summary
                task.setdefault("tail_recovery", {})["published"] = recovered
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
                                                "debug_sheets": args.debug_sheets,
                                                "debug_sheet_max_cameras": args.debug_sheet_max_cameras,
                                                "debug_sheet_output_root_rel": args.debug_sheet_output_root_rel,
                                                "min_valid_pose_coverage": args.min_valid_pose_coverage,
                                                "max_trailing_missing_frames": args.max_trailing_missing_frames,
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
    args.foundpose_selection_mode = str(manifest.get("foundpose_selection_mode", args.foundpose_selection_mode))
    args.foundpose_candidate_rank = int(manifest.get("foundpose_candidate_rank", args.foundpose_candidate_rank))
    args.foundpose_per_view_candidates = int(manifest.get("foundpose_per_view_candidates", args.foundpose_per_view_candidates))
    args.foundpose_global_rotation_count = int(manifest.get("foundpose_global_rotation_count", args.foundpose_global_rotation_count))
    args.foundpose_global_coarse_max_side = int(manifest.get("foundpose_global_coarse_max_side", args.foundpose_global_coarse_max_side))
    args.foundpose_global_refine_top_k = int(manifest.get("foundpose_global_refine_top_k", args.foundpose_global_refine_top_k))
    args.foundpose_global_min_rotation_separation_deg = float(manifest.get("foundpose_global_min_rotation_separation_deg", args.foundpose_global_min_rotation_separation_deg))
    args.foundpose_global_asymmetry_refine_top_k = int(manifest.get("foundpose_global_asymmetry_refine_top_k", args.foundpose_global_asymmetry_refine_top_k))
    args.foundpose_global_asymmetry_max_side = int(manifest.get("foundpose_global_asymmetry_max_side", args.foundpose_global_asymmetry_max_side))
    args.foundpose_global_asymmetry_score_margin = float(manifest.get("foundpose_global_asymmetry_score_margin", args.foundpose_global_asymmetry_score_margin))
    args.foundpose_global_asymmetry_weight = float(manifest.get("foundpose_global_asymmetry_weight", args.foundpose_global_asymmetry_weight))
    args.debug_sheets = bool(manifest.get("debug_sheets", args.debug_sheets))
    args.debug_sheet_max_cameras = int(manifest.get("debug_sheet_max_cameras", args.debug_sheet_max_cameras))
    args.debug_sheet_output_root_rel = str(manifest.get("debug_sheet_output_root_rel", args.debug_sheet_output_root_rel))
    args.min_valid_pose_coverage = float(manifest.get("min_valid_pose_coverage", args.min_valid_pose_coverage))
    args.max_trailing_missing_frames = int(manifest.get("max_trailing_missing_frames", args.max_trailing_missing_frames))
    args.tail_recovery = bool(manifest.get("tail_recovery", args.tail_recovery))
    args.tail_recovery_frame_step = int(manifest.get("tail_recovery_frame_step", args.tail_recovery_frame_step))
    args.tail_recovery_max_seed_attempts = int(manifest.get("tail_recovery_max_seed_attempts", args.tail_recovery_max_seed_attempts))
    args.tail_recovery_min_mask_views = int(manifest.get("tail_recovery_min_mask_views", args.tail_recovery_min_mask_views))
    args.tail_recovery_overlap_frames = int(manifest.get("tail_recovery_overlap_frames", args.tail_recovery_overlap_frames))
    args.tail_recovery_min_connection_frames = int(manifest.get("tail_recovery_min_connection_frames", args.tail_recovery_min_connection_frames))
    args.tail_recovery_min_suffix_coverage = float(manifest.get("tail_recovery_min_suffix_coverage", args.tail_recovery_min_suffix_coverage))
    args.tail_recovery_max_translation_error_m = float(manifest.get("tail_recovery_max_translation_error_m", args.tail_recovery_max_translation_error_m))
    args.tail_recovery_max_rotation_error_deg = float(manifest.get("tail_recovery_max_rotation_error_deg", args.tail_recovery_max_rotation_error_deg))
    args.tail_recovery_max_rotation_alignment_dispersion_deg = float(manifest.get("tail_recovery_max_rotation_alignment_dispersion_deg", args.tail_recovery_max_rotation_alignment_dispersion_deg))
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
                   "--foundpose-selection-mode", args.foundpose_selection_mode,
                   "--foundpose-candidate-rank", str(args.foundpose_candidate_rank),
                   "--foundpose-per-view-candidates", str(args.foundpose_per_view_candidates),
                   "--foundpose-global-rotation-count", str(args.foundpose_global_rotation_count),
                   "--foundpose-global-coarse-max-side", str(args.foundpose_global_coarse_max_side),
                   "--foundpose-global-refine-top-k", str(args.foundpose_global_refine_top_k),
                   "--foundpose-global-min-rotation-separation-deg", str(args.foundpose_global_min_rotation_separation_deg),
                   "--foundpose-global-asymmetry-refine-top-k", str(args.foundpose_global_asymmetry_refine_top_k),
                   "--foundpose-global-asymmetry-max-side", str(args.foundpose_global_asymmetry_max_side),
                   "--foundpose-global-asymmetry-score-margin", str(args.foundpose_global_asymmetry_score_margin),
                   "--foundpose-global-asymmetry-weight", str(args.foundpose_global_asymmetry_weight),
                   "--debug-sheet-max-cameras", str(args.debug_sheet_max_cameras),
                   "--debug-sheet-output-root-rel", args.debug_sheet_output_root_rel,
                   "--min-valid-pose-coverage", str(args.min_valid_pose_coverage),
                   "--max-trailing-missing-frames", str(args.max_trailing_missing_frames),
                   "--tail-recovery-frame-step", str(args.tail_recovery_frame_step),
                   "--tail-recovery-max-seed-attempts", str(args.tail_recovery_max_seed_attempts),
                   "--tail-recovery-min-mask-views", str(args.tail_recovery_min_mask_views),
                   "--tail-recovery-overlap-frames", str(args.tail_recovery_overlap_frames),
                   "--tail-recovery-min-connection-frames", str(args.tail_recovery_min_connection_frames),
                   "--tail-recovery-min-suffix-coverage", str(args.tail_recovery_min_suffix_coverage),
                   "--tail-recovery-max-translation-error-m", str(args.tail_recovery_max_translation_error_m),
                   "--tail-recovery-max-rotation-error-deg", str(args.tail_recovery_max_rotation_error_deg),
                   "--tail-recovery-max-rotation-alignment-dispersion-deg", str(args.tail_recovery_max_rotation_alignment_dispersion_deg),
                   "--max-frames", str(args.max_frames), "--max-attempts", str(args.max_attempts)]
        command.append("--debug-sheets" if args.debug_sheets else "--no-debug-sheets")
        command.append("--tail-recovery" if args.tail_recovery else "--no-tail-recovery")
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
    p.add_argument("--cache-root-rel", default="mesh_new",
                   help="Canonical FoundPose cache root under ~/shared_data. Default: mesh_new")
    p.add_argument("--mesh-root-rel", default="mesh_new",
                   help="Preferred mesh evidence root under ~/shared_data. Default: mesh_new")
    p.add_argument("--fallback-mesh-root-rel", default="mesh_blender",
                   help="Fallback evidence root for objects absent from --mesh-root-rel. Default: mesh_blender")
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
    p.add_argument("--foundpose-selection-mode", choices=("silhouette", "consensus", "hybrid", "global"),
                   default="silhouette",
                   help="FoundPose wrapper selector. global adds coarse SO(3) search and multi-start silhouette refinement.")
    p.add_argument("--foundpose-candidate-rank", type=int, default=0,
                   help="Zero-based candidate-bank rank to initialize GoTrack from. Default: 0.")
    p.add_argument("--foundpose-per-view-candidates", type=int, default=5,
                   help="PnP alternatives retained per camera in global mode. Default: 5.")
    p.add_argument("--foundpose-global-rotation-count", type=int, default=256)
    p.add_argument("--foundpose-global-coarse-max-side", type=int, default=160)
    p.add_argument("--foundpose-global-refine-top-k", type=int, default=5)
    p.add_argument("--foundpose-global-min-rotation-separation-deg", type=float, default=35.0)
    p.add_argument("--foundpose-global-asymmetry-refine-top-k", type=int, default=12)
    p.add_argument("--foundpose-global-asymmetry-max-side", type=int, default=512)
    p.add_argument("--foundpose-global-asymmetry-score-margin", type=float, default=0.005)
    p.add_argument("--foundpose-global-asymmetry-weight", type=float, default=0.7)
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
    recovery_group = p.add_mutually_exclusive_group()
    recovery_group.add_argument("--tail-recovery", dest="tail_recovery", action="store_true",
                                help="Re-anchor from the end and reverse-track a lost terminal suffix (default).")
    recovery_group.add_argument("--no-tail-recovery", dest="tail_recovery", action="store_false")
    p.set_defaults(tail_recovery=True)
    p.add_argument("--tail-recovery-frame-step", type=int, default=30,
                   help="Move this many frames toward the front after an unusable late seed.")
    p.add_argument("--tail-recovery-max-seed-attempts", type=int, default=6)
    p.add_argument("--tail-recovery-min-mask-views", type=int, default=6)
    p.add_argument("--tail-recovery-overlap-frames", type=int, default=30)
    p.add_argument("--tail-recovery-min-connection-frames", type=int, default=3)
    p.add_argument("--tail-recovery-min-suffix-coverage", type=float, default=0.9)
    p.add_argument("--tail-recovery-max-translation-error-m", type=float, default=0.03)
    p.add_argument("--tail-recovery-max-rotation-error-deg", type=float, default=15.0)
    p.add_argument("--tail-recovery-max-rotation-alignment-dispersion-deg", type=float, default=5.0,
                   help="Allow a constant late-seed rotation offset only below this overlap dispersion.")
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-workers-stopped", action="store_true",
                   help="Required with --mode reset-running; confirms no worker can still own a task.")
    args = p.parse_args()
    if (args.connect_timeout < 1 or args.num_cameras < 1 or args.camera_micro_batch_size < 0
            or args.max_video_duration_skew_sec < 0 or args.max_frames == 0 or args.max_attempts < 1
            or args.init_frame_index < 0 or args.foundpose_candidate_rank < 0
            or args.foundpose_per_view_candidates < 1
            or args.foundpose_global_rotation_count < 1 or args.foundpose_global_coarse_max_side < 32
            or args.foundpose_global_refine_top_k < 1 or args.foundpose_global_min_rotation_separation_deg < 0
            or args.foundpose_global_asymmetry_refine_top_k < args.foundpose_global_refine_top_k
            or args.foundpose_global_asymmetry_max_side < 32 or args.foundpose_global_asymmetry_score_margin < 0
            or not 0 <= args.foundpose_global_asymmetry_weight <= 1 or args.debug_sheet_max_cameras < 1
            or not 0 < args.min_valid_pose_coverage <= 1 or args.max_trailing_missing_frames < 0
            or args.tail_recovery_frame_step < 1 or args.tail_recovery_max_seed_attempts < 1
            or args.tail_recovery_min_mask_views < 1 or args.tail_recovery_overlap_frames < 1
            or args.tail_recovery_min_connection_frames < 1
            or not 0 < args.tail_recovery_min_suffix_coverage <= 1
            or args.tail_recovery_max_translation_error_m < 0
            or args.tail_recovery_max_rotation_error_deg < 0
            or args.tail_recovery_max_rotation_alignment_dispersion_deg < 0):
        raise ValueError("invalid worker/tracking option")
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
