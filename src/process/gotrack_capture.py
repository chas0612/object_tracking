#!/usr/bin/env python3
"""Track one capture archive with GoTrack from a FoundPose initialization.

This adapter never changes existing capture inputs or earlier pipeline output.
By default it creates a new ``gotrack_tracking/`` directory under
``--capture-dir`` and writes the staged GoTrack input, generated anchor bank,
run manifest, and tracker output there only.

Run inside the ``gotrack`` conda environment.  The default is a 100-frame
smoke test; pass ``--max-frames -1`` only after checking the result.

Example:
    conda run --no-capture-output -n gotrack python -u \
      src/process/gotrack_capture.py \
      --capture-dir /path/to/apple/0 \
      --mesh /path/to/apple.obj \
      --init-pose /path/to/foundpose_init/init_pose_world.npy \
      --object-name apple --num-cameras 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GOTRACK_ROOT = REPO_ROOT / "autodex" / "perception" / "thirdparty" / "MV-GoTrack"
GOTRACK_RUNNER = GOTRACK_ROOT / "run_multiview_gotrack_anchor_online_multi_object.py"
ANCHOR_GENERATOR = GOTRACK_ROOT / "scripts" / "generate_anchor_bank.py"


def _as_pose_4x4(path: Path) -> np.ndarray:
    pose = np.load(path)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0.0, 0.0, 0.0, 1.0]])
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Expected a finite 3x4 or 4x4 pose in {path}, got {pose.shape}")
    return pose.astype(np.float64)


def _video_timing(video_path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return {"frames": 0, "fps": 0.0, "duration_sec": 0.0}
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        # AVI has no independently useful per-frame capture timestamps exposed
        # through OpenCV.  Frame count / encoded FPS is its usable timeline.
        return {"frames": frames, "fps": fps, "duration_sec": frames / fps if fps > 0 else 0.0}
    finally:
        cap.release()


def _eligible_cameras(
    videos_dir: Path, min_frames: int, excluded: set[str], max_duration_skew_sec: float,
) -> tuple[list[str], dict[str, dict[str, float | int]], dict[str, str], float | None]:
    """Keep normal-length videos and reject timeline outliers across cameras.

    A few dropped frames are expected from the capture system.  We compare the
    encoded video duration (frame_count / FPS) to the median of otherwise valid
    cameras, rather than excluding a historical hard-coded serial.
    """
    timings = {p.stem: _video_timing(p) for p in sorted(videos_dir.glob("*.avi"))}
    rejected: dict[str, str] = {}
    baseline: list[float] = []
    for camera_id, info in timings.items():
        frames, fps, duration = int(info["frames"]), float(info["fps"]), float(info["duration_sec"])
        if camera_id in excluded:
            rejected[camera_id] = "manual_exclude"
        elif frames < min_frames:
            rejected[camera_id] = f"frames<{min_frames}"
        elif fps <= 0 or duration <= 0:
            rejected[camera_id] = "invalid_fps_or_duration"
        else:
            baseline.append(duration)
    median_duration = statistics.median(baseline) if baseline else None
    eligible: list[str] = []
    for camera_id, info in timings.items():
        if camera_id in rejected:
            continue
        duration = float(info["duration_sec"])
        skew = abs(duration - median_duration) if median_duration is not None else 0.0
        info["duration_skew_sec"] = skew
        if max_duration_skew_sec > 0 and skew > max_duration_skew_sec:
            rejected[camera_id] = f"duration_skew={skew:.3f}s>{max_duration_skew_sec:.3f}s"
        else:
            eligible.append(camera_id)
    return sorted(eligible), timings, rejected, median_duration


def _select_cameras(extrinsics_json: Path, available: list[str], number: int) -> list[str]:
    cmd = [
        sys.executable, str(GOTRACK_ROOT / "scripts" / "select_cameras.py"),
        "--extrinsics-json", str(extrinsics_json),
        "--num-cameras", str(number), "--mode", "fps",
        "--available-ids", *available,
    ]
    result = subprocess.run(cmd, check=True, cwd=str(GOTRACK_ROOT), text=True, capture_output=True)
    selected = result.stdout.strip().split()
    if len(selected) != number:
        raise RuntimeError(f"Camera selector returned {len(selected)} cameras, expected {number}: {selected}")
    return selected


def _write_init_pose_jsons(
    init_dir: Path, camera_ids: list[str], pose: np.ndarray, frame_index: int,
) -> None:
    frame_poses = init_dir / "frame_poses"
    frame_poses.mkdir(parents=True)
    record = [{
        "frame_index": int(frame_index),
        "pose_world": pose.tolist(),
        "certainty_count_above_threshold": 1000.0,
        "status": "ok",
    }]
    for camera_id in camera_ids:
        with open(frame_poses / f"{camera_id}.json", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)


def _stage_input(capture_dir: Path, video_dir: Path, stage_dir: Path) -> None:
    stage_dir.mkdir()
    for name in ("cam_param",):
        source = capture_dir / name
        if not source.is_dir():
            raise FileNotFoundError(f"Missing required capture directory: {source}")
        (stage_dir / name).symlink_to(source.resolve(), target_is_directory=True)
    # MV-GoTrack prioritizes undistorted_video/.  Always stage the selected
    # input under that name so it cannot silently fall back to raw videos/.
    (stage_dir / "undistorted_video").symlink_to(video_dir.resolve(), target_is_directory=True)


def _write_reversed_prefix_videos(
    source_dir: Path, destination_dir: Path, camera_ids: list[str], frame_index: int,
    stop_frame_index: int = 0,
) -> None:
    """Write frames ``frame_index..stop_frame_index`` for the backward pass.

    We deliberately do not reverse the whole capture: the backward pass only
    needs to recover the short pre-seed prefix.  ``MJPG`` is used because it
    is reliably readable by OpenCV and MV-GoTrack on all capture workers.
    """
    destination_dir.mkdir(parents=True)
    for camera_id in camera_ids:
        source = source_dir / f"{camera_id}.avi"
        cap = cv2.VideoCapture(str(source))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video for reverse pass: {source}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_index >= total:
                raise ValueError(f"--init-frame-index {frame_index} is outside {source} (0..{total - 1})")
            if fps <= 0 or width <= 0 or height <= 0:
                raise RuntimeError(f"Invalid AVI metadata for reverse pass: {source}")
            target = destination_dir / source.name
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not create reverse video: {target}")
            try:
                for original_index in range(frame_index, stop_frame_index - 1, -1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, original_index)
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Could not decode {source} frame {original_index} for reverse pass")
                    writer.write(frame)
            finally:
                writer.release()
        finally:
            cap.release()


def _load_records(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a list of records in {path}")
    return payload


def _merge_bidirectional_records(
    forward_path: Path, backward_path: Path, destination: Path, seed_frame: int,
) -> dict[str, int]:
    """Map reverse-local frames back to capture indices and join at the seed.

    The FoundPose seed belongs to the forward run.  Earlier frames come from
    the reversed prefix, so the public output remains one normal chronological
    ``world_pose_records.json`` usable by all existing visualization tools.
    """
    forward = _load_records(forward_path)
    backward = _load_records(backward_path)
    merged = {int(row["frame_index"]): dict(row) for row in forward if "frame_index" in row}
    backward_used = 0
    for reverse_row in backward:
        if "frame_index" not in reverse_row:
            continue
        original_index = seed_frame - int(reverse_row["frame_index"])
        if original_index < 0 or original_index >= seed_frame:
            continue
        row = dict(reverse_row)
        row["frame_index"] = original_index
        row["tracking_direction"] = "backward"
        row["reverse_source_frame_index"] = int(reverse_row["frame_index"])
        merged[original_index] = row
        backward_used += 1
    ordered: list[dict[str, object]] = []
    for frame_index in sorted(merged):
        row = merged[frame_index]
        row.setdefault("tracking_direction", "forward")
        ordered.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    return {"forward_records": len(forward), "backward_records_used": backward_used, "merged_records": len(ordered)}


def _require_gotrack_dependencies() -> None:
    missing = [name for name in ("hydra", "omegaconf") if importlib.util.find_spec(name) is None]
    if missing:
        packages = " ".join("hydra-core==1.3.2" if name == "hydra" else "omegaconf" for name in missing)
        raise RuntimeError(
            "The active environment is missing GoTrack dependencies: " + ", ".join(missing) + ".\n"
            f"Install them in the gotrack environment, for example:\n  pip install {packages}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", required=True, help="Capture archive with cam_param/.")
    parser.add_argument("--video-dir", default=None,
                        help="Input AVI directory. Default: <capture-dir>/undistorted_video (required).")
    parser.add_argument("--mesh", required=True, help="Object mesh in meters.")
    parser.add_argument("--init-pose", required=True, help="FoundPose init_pose_world.npy.")
    parser.add_argument("--init-frame-index", type=int, default=30,
                        help="Video frame represented by --init-pose. Default 30 avoids known stale frame-0 captures.")
    parser.add_argument("--reverse-stop-frame-index", type=int, default=0,
                        help=("Earliest original frame written for the reverse pass. "
                              "A positive value limits late re-anchor recovery to a short overlap."))
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument("--bidirectional", dest="bidirectional", action="store_true",
                                 help="Track forward and reverse the 0..seed prefix, then merge into one chronological output (default).")
    direction_group.add_argument("--forward-only", dest="bidirectional", action="store_false",
                                 help="Only track forward from the seed; frames before it remain untracked.")
    parser.set_defaults(bidirectional=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--output-dir", default=None,
                        help="Default: <capture-dir>/gotrack_tracking/ (new directory only).")
    parser.add_argument("--num-cameras", type=int, default=4, help="FPS-selected cameras to track.")
    parser.add_argument("--allow-fewer-cameras", action="store_true",
                        help="When automatic filtering leaves fewer than --num-cameras, track all eligible cameras instead of failing.")
    parser.add_argument("--camera-ids", nargs="*", default=None,
                        help="Use these camera IDs instead of FPS selection.")
    parser.add_argument("--gpus", nargs="+", default=["0"],
                        help="CUDA device IDs for camera-group sharding, e.g. --gpus 0 1.")
    parser.add_argument("--exclude-cameras", nargs="*", default=[],
                        help="Optional camera IDs to exclude explicitly. Default: none.")
    parser.add_argument("--min-video-frames", type=int, default=100,
                        help="Reject videos shorter than this before selection.")
    parser.add_argument("--max-video-duration-skew-sec", type=float, default=1.0,
                        help="Reject videos whose frame_count/FPS duration differs from the valid-camera median by more than this. 0 disables it.")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Tracking length; default is a safe 100-frame smoke test, -1 means all frames.")
    parser.add_argument("--input-resize-scale", type=float, default=0.5)
    parser.add_argument("--first-frame-num-iters", type=int, default=5)
    parser.add_argument("--num-anchors", type=int, default=256)
    parser.add_argument("--camera-micro-batch-size", type=int, default=0,
                        help="GPU views per sequential refinement micro-batch; 0 keeps all selected views together.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the run manifest without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    video_dir = (Path(args.video_dir).expanduser().resolve() if args.video_dir else
                 capture_dir / "undistorted_video")
    mesh_path = Path(args.mesh).expanduser().resolve()
    init_pose_path = Path(args.init_pose).expanduser().resolve()
    if not capture_dir.is_dir() or not video_dir.is_dir() or not mesh_path.is_file() or not init_pose_path.is_file():
        raise FileNotFoundError("--capture-dir/--video-dir must exist, and --mesh/--init-pose must be files.")
    if not GOTRACK_RUNNER.is_file():
        raise FileNotFoundError(f"GoTrack runner not found: {GOTRACK_RUNNER}")
    if (args.num_cameras < 1 or args.min_video_frames < 1 or args.camera_micro_batch_size < 0
            or args.max_video_duration_skew_sec < 0 or args.init_frame_index < 0
            or not 0 <= args.reverse_stop_frame_index <= args.init_frame_index):
        raise ValueError("--num-cameras/--min-video-frames must be positive, and batch/skew options must be non-negative.")

    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
                  capture_dir / "gotrack_tracking")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to modify an existing output directory: {output_dir}")

    pose = _as_pose_4x4(init_pose_path)
    excluded = set(args.exclude_cameras)
    eligible, video_timings, rejected_cameras, median_duration = _eligible_cameras(
        video_dir, args.min_video_frames, excluded, args.max_video_duration_skew_sec,
    )
    requested = list(args.camera_ids) if args.camera_ids else None
    if requested:
        selected = sorted(set(requested))
        rejected = [camera_id for camera_id in selected if camera_id not in eligible]
        if rejected:
            raise ValueError(f"Requested cameras are excluded, missing, short, or duration outliers: {rejected}")
    else:
        if len(eligible) < args.num_cameras:
            if not args.allow_fewer_cameras:
                raise RuntimeError(f"Only {len(eligible)} eligible cameras, need {args.num_cameras}: {eligible}; rejected={rejected_cameras}")
            print(f"[camera-filter] requested={args.num_cameras}, eligible={len(eligible)}; using all eligible cameras", flush=True)
        selected = _select_cameras(
            capture_dir / "cam_param" / "extrinsics.json", eligible, min(args.num_cameras, len(eligible)),
        )

    if args.init_frame_index >= min(int(video_timings[camera_id]["frames"]) for camera_id in selected):
        raise ValueError(f"--init-frame-index {args.init_frame_index} is outside at least one selected camera video")
    manifest = {
        "capture_dir": str(capture_dir), "video_dir": str(video_dir), "mesh": str(mesh_path), "init_pose": str(init_pose_path),
        "object_name": args.object_name, "requested_num_cameras": args.num_cameras,
        "allow_fewer_cameras": args.allow_fewer_cameras, "selected_cameras": selected,
        "manual_excluded_cameras": sorted(excluded), "rejected_cameras": rejected_cameras,
        "video_timings": video_timings, "median_video_duration_sec": median_duration,
        "min_video_frames": args.min_video_frames, "max_video_duration_skew_sec": args.max_video_duration_skew_sec,
        "max_frames": args.max_frames,
        "init_frame_index": args.init_frame_index,
        "reverse_stop_frame_index": args.reverse_stop_frame_index,
        "bidirectional": args.bidirectional,
        "gpus": args.gpus, "camera_micro_batch_size": args.camera_micro_batch_size,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    _require_gotrack_dependencies()
    output_dir.mkdir(parents=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    forward_stage_dir = output_dir / "forward_input_stage"
    forward_init_dir = output_dir / "forward_init_poses"
    anchor_bank = output_dir / "anchor_bank.npz"
    forward_output = output_dir / "forward_output"
    reverse_video_dir = output_dir / "reverse_prefix_videos"
    reverse_stage_dir = output_dir / "reverse_input_stage"
    reverse_init_dir = output_dir / "reverse_init_poses"
    reverse_output = output_dir / "reverse_output"
    final_records = output_dir / "gotrack_output" / args.object_name / "world_pose_records.json"
    _stage_input(capture_dir, video_dir, forward_stage_dir)
    _write_init_pose_jsons(forward_init_dir, selected, pose, args.init_frame_index)

    generate_anchor_cmd = [
        sys.executable, str(ANCHOR_GENERATOR), "--mesh-path", str(mesh_path),
        "--output-path", str(anchor_bank), "--num-anchors", str(args.num_anchors), "--mesh-scale", "1.0",
    ]
    subprocess.run(generate_anchor_cmd, check=True, cwd=str(GOTRACK_ROOT))
    track_common = [
        sys.executable, str(GOTRACK_RUNNER),
        "--checkpoint-path", str(GOTRACK_ROOT / "gotrack_checkpoint.pt"), "--gpus", *args.gpus,
        "--camera-ids", *selected, "--object-names", args.object_name, "--object-ids", "1",
        "--mesh-paths", str(mesh_path),
        "--anchor-bank-paths", str(anchor_bank), "--num-iters", "1",
        "--first-frame-num-iters", str(args.first_frame_num_iters), "--num-anchors", str(args.num_anchors),
        "--mesh-scale", "1.0", "--unit-scale-mode", "auto", "--mask-free", "--skip-pnp",
        "--optimized-input-pipeline-v2", "--optim-v2-crop-camera-workers", "4",
        "--optim-v2-warp-grid-workers", "4", "--optim-template-update-interval", "2",
        "--template-renderer-backend", "nvdiffrast", "--input-resize-scale", str(args.input_resize_scale),
        "--camera-micro-batch-size", str(args.camera_micro_batch_size),
        "--forward-precision", "fp32", "--torch-compile", "off",
        "--worker-mode", "auto", "--tri-fit-worker-mode", "process", "--triangulation-worker-mode", "auto",
        "--status-log-every", "50", "--debug-level", "0",
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless")
    print("[gotrack] selected cameras:", " ".join(selected), flush=True)
    print("[gotrack] output:", output_dir, flush=True)
    forward_cmd = track_common + [
        "--input-root", str(forward_stage_dir), "--output-root", str(forward_output),
        "--init-pose-sources", str(forward_init_dir), "--max-frames", str(args.max_frames),
    ]
    subprocess.run(forward_cmd, check=True, cwd=str(GOTRACK_ROOT), env=env)
    forward_records = forward_output / args.object_name / "world_pose_records.json"
    if not args.bidirectional or args.init_frame_index == 0:
        final_records.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(forward_records, final_records)
        print(f"[done] {final_records}", flush=True)
        return 0

    try:
        print(f"[gotrack] building reversed interval "
              f"{args.reverse_stop_frame_index}..{args.init_frame_index}", flush=True)
        _write_reversed_prefix_videos(
            video_dir, reverse_video_dir, selected, args.init_frame_index,
            stop_frame_index=args.reverse_stop_frame_index,
        )
        _stage_input(capture_dir, reverse_video_dir, reverse_stage_dir)
        _write_init_pose_jsons(reverse_init_dir, selected, pose, 0)
        reverse_cmd = track_common + [
            "--input-root", str(reverse_stage_dir), "--output-root", str(reverse_output),
            "--init-pose-sources", str(reverse_init_dir), "--max-frames", str(args.init_frame_index + 1),
        ]
        subprocess.run(reverse_cmd, check=True, cwd=str(GOTRACK_ROOT), env=env)
        merge_stats = _merge_bidirectional_records(
            forward_records, reverse_output / args.object_name / "world_pose_records.json", final_records, args.init_frame_index,
        )
        (output_dir / "merge_manifest.json").write_text(json.dumps({
            "seed_frame": args.init_frame_index,
            "reverse_stop_frame": args.reverse_stop_frame_index,
            "strategy": "forward_from_seed_plus_reverse_interval", **merge_stats,
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        # This is a reproducible, short-lived staging artifact.  Keep tracker
        # outputs for audit, but avoid permanently storing 22 duplicate AVIs.
        if reverse_video_dir.exists():
            shutil.rmtree(reverse_video_dir)
    print(f"[done] {final_records}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
