#!/usr/bin/env python3
"""Build a write-isolated workspace from complete Robotiq grasp failures.

An episode is eligible only when the capture reached its final grasp-result
write, all calibrated videos and robot raw arrays are intact, its paired human
episode is a real low-numbered dataset episode, and the recorded xArm motion
does not violate or remain saturated at a URDF joint limit.

This is a conservative legacy audit, not the final tracking-eligibility rule.
In particular, scalar teleop recordings can be intentional for gripper captures;
a teleop rejection requires manual re-audit and does not imply broken arm/video
data.  Paired human numbering also requires checking the capture correspondence,
not merely assuming that it equals the exported human episode number.

The source capture is never modified.  The default is a read-only audit; pass
``--write`` to create relative symlinks and per-episode provenance manifests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_SHARED_ROOT = Path.home() / "shared_data"
DEFAULT_SOURCE_REL = Path("capture/eccv2026/robotiq_2f85")
DEFAULT_HUMAN_REL = Path("capture/eccv2026/v0/human")
DEFAULT_WORKSPACE_REL = Path("object_tracking/campaigns/robotiq_2f85_failed/workspace")

REQUIRED_LINKED_NAMES = (
    "cam_param", "C2R.npy", "raw", "videos", "grasp_result.json",
    "paired_human_episode.json",
)
RAW_GROUPS = {
    "arm": ("action.npy", "action_qpos.npy", "position.npy", "time.npy", "torque.npy", "velocity.npy"),
    "hand": ("action.npy", "command.npy", "position.npy", "time.npy"),
    "state": ("state_hist.npy", "state_time.npy"),
    "teleop": ("left.npy", "right.npy", "time.npy"),
    "timestamps": ("frame_id.npy", "timestamp.npy"),
}

# Limits from the xArm6 joints in the robot URDF.  The gripper joints are not
# used here because the requested exclusion concerns the xArm motion.
XARM_LOWER = np.asarray([-2 * math.pi, -2.059, -3.927, -2 * math.pi, -1.69297, -2 * math.pi])
XARM_UPPER = np.asarray([2 * math.pi, 2.0944, 0.19198, 2 * math.pi, math.pi, 2 * math.pi])


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _relative_target(link: Path, target: Path) -> str:
    return os.path.relpath(target, start=link.parent)


def _ensure_relative_symlink(link: Path, target: Path, write: bool) -> None:
    expected = _relative_target(link, target)
    if link.is_symlink():
        if os.readlink(link) != expected or not link.exists():
            raise ValueError(f"Unexpected or broken symlink: {link}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    if write:
        link.symlink_to(expected, target_is_directory=target.is_dir())


def _load_array(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=True)
    if value.size == 0:
        raise ValueError("empty array")
    return value


def _max_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _raw_audit(raw: Path, margin_deg: float, saturation_samples: int) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}
    for group, names in RAW_GROUPS.items():
        group_lengths: set[int] = set()
        for name in names:
            rel = f"{group}/{name}"
            path = raw / rel
            if not path.is_file():
                raise FileNotFoundError(rel)
            array = _load_array(path)
            if array.ndim == 0:
                raise ValueError(f"{rel} is a scalar, expected a time sequence")
            arrays[rel] = array
            group_lengths.add(len(array))
        if len(group_lengths) != 1:
            raise ValueError(f"{group} array lengths disagree: {sorted(group_lengths)}")
        lengths[group] = group_lengths.pop()

    qpos = np.asarray(arrays["arm/position.npy"], dtype=np.float64)
    command = np.asarray(arrays["arm/action_qpos.npy"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 6 or command.shape != qpos.shape:
        raise ValueError(f"invalid xArm shapes: position={qpos.shape}, command={command.shape}")
    if not np.isfinite(qpos).all() or not np.isfinite(command).all():
        raise ValueError("non-finite xArm position/command")

    measured_margin = np.minimum(qpos - XARM_LOWER, XARM_UPPER - qpos)
    command_margin = np.minimum(command - XARM_LOWER, XARM_UPPER - command)
    margin = np.deg2rad(margin_deg)
    near = measured_margin <= margin
    longest = max(_max_true_run(near[:, joint]) for joint in range(6))
    hard_violation = bool((measured_margin < -1e-4).any() or (command_margin < -1e-4).any())
    saturated = longest >= saturation_samples

    frame_ids = np.asarray(arrays["timestamps/frame_id.npy"], dtype=np.int64)
    timestamps = np.asarray(arrays["timestamps/timestamp.npy"], dtype=np.float64)
    if frame_ids.ndim != 1 or timestamps.ndim != 1 or len(frame_ids) != len(timestamps):
        raise ValueError("invalid timestamp arrays")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps are non-finite or non-monotonic")
    if len(frame_ids) > 1 and not np.all(np.diff(frame_ids) == 1):
        raise ValueError("frame IDs are not contiguous")

    return {
        "raw_lengths": lengths,
        "timestamp_frames": int(len(frame_ids)),
        "xarm_min_measured_margin_deg": float(np.rad2deg(measured_margin.min())),
        "xarm_min_command_margin_deg": float(np.rad2deg(command_margin.min())),
        "xarm_near_limit_longest_run": int(longest),
        "xarm_hard_limit_violation": hard_violation,
        "xarm_limit_saturated": saturated,
    }


def _video_timing(path: Path) -> tuple[str, dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("video cannot be opened")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frames <= 0 or fps <= 0:
            raise ValueError(f"invalid frames/fps: {frames}/{fps}")
        return path.stem, {
            "frames": frames,
            "fps": fps,
            "duration_sec": frames / fps,
        }
    finally:
        capture.release()


def _video_audit(
    videos: list[Path], timestamp_frames: int, max_duration_skew_sec: float,
    max_frame_count_difference: int, probe_workers: int,
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=probe_workers) as pool:
        timings = dict(pool.map(_video_timing, videos))
    durations = np.asarray([float(value["duration_sec"]) for value in timings.values()])
    frames = np.asarray([int(value["frames"]) for value in timings.values()])
    median_duration = float(np.median(durations))
    max_skew = float(np.max(np.abs(durations - median_duration)))
    max_frame_delta = int(np.max(np.abs(frames - timestamp_frames)))
    if max_skew > max_duration_skew_sec:
        raise ValueError(
            f"video duration skew {max_skew:.3f}s exceeds {max_duration_skew_sec:.3f}s"
        )
    if max_frame_delta > max_frame_count_difference:
        raise ValueError(
            f"video/timestamp frame delta {max_frame_delta} exceeds {max_frame_count_difference}"
        )
    return {
        "video_min_frames": int(frames.min()),
        "video_max_frames": int(frames.max()),
        "video_median_duration_sec": median_duration,
        "video_max_duration_skew_sec": max_skew,
        "video_max_timestamp_frame_delta": max_frame_delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--source-rel", type=Path, default=DEFAULT_SOURCE_REL)
    parser.add_argument("--human-root-rel", type=Path, default=DEFAULT_HUMAN_REL)
    parser.add_argument("--workspace-rel", type=Path, default=DEFAULT_WORKSPACE_REL)
    parser.add_argument("--max-paired-human-episode", type=int, default=6)
    parser.add_argument("--allowed-camera-counts", nargs="+", type=int, default=(22, 23))
    parser.add_argument("--max-duration-skew-sec", type=float, default=1.0)
    parser.add_argument("--max-frame-count-difference", type=int, default=6)
    parser.add_argument(
        "--joint-limit-margin-deg", type=float, default=2.0,
        help="Controller safety margin used to detect sustained xArm limit saturation.",
    )
    parser.add_argument("--joint-limit-saturation-samples", type=int, default=10)
    parser.add_argument("--video-probe-workers", type=int, default=8)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for value in (args.source_rel, args.human_root_rel, args.workspace_rel):
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("Dataset roots must be safe relative paths")
    if (
        args.max_paired_human_episode < 0
        or args.max_duration_skew_sec < 0
        or args.max_frame_count_difference < 0
        or args.joint_limit_margin_deg < 0
        or args.joint_limit_saturation_samples < 1
        or args.video_probe_workers < 1
    ):
        raise ValueError("Thresholds and worker counts are invalid")

    shared = args.shared_root.expanduser().resolve()
    source = shared / args.source_rel
    human_root = shared / args.human_root_rel
    workspace = shared / args.workspace_rel
    if not source.is_dir() or not human_root.is_dir():
        raise FileNotFoundError("Source Robotiq or paired-human root is missing")

    rows: list[dict[str, Any]] = []
    eligible_rows: list[tuple[Path, dict[str, Any]]] = []
    labelled_failures = missing_results = successful = 0
    reason_histogram: dict[str, int] = {}

    for episode in sorted(path for path in source.glob("*/*") if path.is_dir()):
        tag = f"{episode.parent.name}/{episode.name}"
        grasp_path = episode / "grasp_result.json"
        if not grasp_path.is_file():
            missing_results += 1
            continue
        grasp = _read_json(grasp_path)
        if bool(grasp.get("grasp_success")):
            successful += 1
            continue
        labelled_failures += 1
        row: dict[str, Any] = {
            "object_name": episode.parent.name,
            "episode": episode.name,
            "source_episode_rel": str(episode.relative_to(shared)),
            "status": "eligible",
            "reasons": [],
        }

        try:
            if grasp.get("episode") != int(episode.name):
                raise ValueError("grasp_result episode does not match directory")
            pair_path = episode / "paired_human_episode.json"
            if not pair_path.is_file():
                raise FileNotFoundError("paired_human_episode.json")
            pair = _read_json(pair_path)
            paired = pair.get("paired human episode")
            if not isinstance(paired, int) or not 0 <= paired <= args.max_paired_human_episode:
                raise ValueError(f"invalid paired human episode: {paired!r}")
            paired_dir = human_root / episode.parent.name / str(paired)
            if not paired_dir.is_dir():
                raise FileNotFoundError(f"paired human target: {paired_dir}")
            row.update({
                "paired_human_episode": paired,
                "paired_human_episode_rel": str(paired_dir.relative_to(shared)),
            })

            for name in REQUIRED_LINKED_NAMES:
                if not (episode / name).exists():
                    raise FileNotFoundError(name)
            c2r = np.asarray(np.load(episode / "C2R.npy"), dtype=np.float64)
            if c2r.shape != (4, 4) or not np.isfinite(c2r).all():
                raise ValueError(f"invalid C2R shape/value: {c2r.shape}")

            intrinsics = _read_json(episode / "cam_param/intrinsics.json")
            extrinsics = _read_json(episode / "cam_param/extrinsics.json")
            camera_ids = sorted(intrinsics)
            if set(intrinsics) != set(extrinsics) or len(camera_ids) not in args.allowed_camera_counts:
                raise ValueError("calibration camera sets/count are invalid")
            videos = sorted((episode / "videos").glob("*.avi"))
            if [path.stem for path in videos] != camera_ids:
                raise ValueError("calibrated camera and AVI sets differ")
            if any(path.stat().st_size == 0 for path in videos):
                raise ValueError("one or more AVI files are empty")

            raw_result = _raw_audit(
                episode / "raw", args.joint_limit_margin_deg,
                args.joint_limit_saturation_samples,
            )
            row.update(raw_result)
            if raw_result["xarm_hard_limit_violation"]:
                raise ValueError("xArm hard joint-limit violation")
            if raw_result["xarm_limit_saturated"]:
                raise ValueError(
                    f"xArm stayed within {args.joint_limit_margin_deg}deg of a joint limit "
                    f"for >= {args.joint_limit_saturation_samples} samples"
                )

            row.update(_video_audit(
                videos, raw_result["timestamp_frames"], args.max_duration_skew_sec,
                args.max_frame_count_difference, args.video_probe_workers,
            ))
            row["camera_count"] = len(camera_ids)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            row.update(status="excluded", reasons=[reason])
            category = reason.split(":", 1)[0] + ":" + reason.split(":", 1)[1].strip().split(":", 1)[0]
            reason_histogram[category] = reason_histogram.get(category, 0) + 1

        rows.append(row)
        if row["status"] == "eligible":
            eligible_rows.append((episode, row))

    for episode, row in eligible_rows:
        destination = workspace / episode.parent.name / episode.name
        if args.write:
            destination.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_LINKED_NAMES:
            _ensure_relative_symlink(destination / name, episode / name, args.write)
        manifest = {
            "schema_version": 1,
            "source_episode_rel": row["source_episode_rel"],
            "workspace_episode_rel": str(destination.relative_to(shared)),
            "object_name": row["object_name"],
            "episode": row["episode"],
            "grasp_success": False,
            "selection_audit": row,
        }
        manifest_path = destination / "source_episode.json"
        if manifest_path.exists():
            if _read_json(manifest_path) != manifest:
                raise ValueError(f"Workspace manifest mismatch: {manifest_path}")
        elif args.write:
            _atomic_json(manifest_path, manifest)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "write" if args.write else "dry-run",
        "source_root": str(source),
        "paired_human_root": str(human_root),
        "workspace_root": str(workspace),
        "counts": {
            "successful": successful,
            "missing_grasp_result": missing_results,
            "labelled_grasp_failures": labelled_failures,
            "eligible_complete_failures": len(eligible_rows),
            "excluded_failures": labelled_failures - len(eligible_rows),
        },
        "thresholds": {
            "max_paired_human_episode": args.max_paired_human_episode,
            "allowed_camera_counts": args.allowed_camera_counts,
            "max_duration_skew_sec": args.max_duration_skew_sec,
            "max_frame_count_difference": args.max_frame_count_difference,
            "joint_limit_margin_deg": args.joint_limit_margin_deg,
            "joint_limit_saturation_samples": args.joint_limit_saturation_samples,
        },
        "exclusion_reason_histogram": reason_histogram,
        "episodes": rows,
        "source_dataset_modified": False,
    }
    print(json.dumps({
        "mode": report["mode"],
        "counts": report["counts"],
        "exclusion_reason_histogram": reason_histogram,
    }, indent=2), flush=True)
    if args.report is not None:
        _atomic_json(args.report.expanduser().resolve(), report)
        print(f"report={args.report.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
