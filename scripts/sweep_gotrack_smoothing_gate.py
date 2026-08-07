#!/usr/bin/env python3
"""Audit motion-gated Gaussian eligibility over selected GoTrack trajectories.

This is read-only. By default it scans promoted ``object_6d_pose_v2.npz`` files,
evaluates one or more translation/angular-speed thresholds, and reports how
often a complete Gaussian window would be safe to apply. It can also inspect a
``latest_trials.json`` manifest, though those detailed JSONs are much heavier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MANIFEST = (
    Path.home()
    / "shared_data/object_tracking/gotrack_debug_sheets/inspire_dftp_latest/latest_trials.json"
)
DEFAULT_TARGET_ROOT = Path.home() / "shared_data/capture/eccv2026/v0/inspire_dftp"


def _load_poses(path: Path) -> np.ndarray:
    rows = json.loads(path.read_text(encoding="utf-8"))
    indexed: list[tuple[int, np.ndarray]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("pose_world") is None:
            continue
        pose = np.asarray(row["pose_world"], dtype=np.float64)
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            indexed.append((int(row["frame_index"]), pose))
    indexed.sort(key=lambda item: item[0])
    if len(indexed) < 2:
        raise ValueError("fewer than two valid poses")
    frames = [item[0] for item in indexed]
    if frames != list(range(frames[0], frames[0] + len(frames))):
        raise ValueError("pose frames are not contiguous")
    return np.asarray([item[1] for item in indexed])


def _load_npz_poses(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        indexed: list[tuple[int, np.ndarray]] = []
        for key in archive.files:
            if not key.startswith("frame_"):
                raise ValueError(f"unexpected NPZ key {key!r}")
            pose = np.asarray(archive[key], dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"invalid pose under {key}")
            indexed.append((int(key.removeprefix("frame_")), pose))
    indexed.sort(key=lambda item: item[0])
    if len(indexed) < 2 or [item[0] for item in indexed] != list(range(len(indexed))):
        raise ValueError("NPZ pose frames are not contiguous from zero")
    return np.asarray([item[1] for item in indexed])


def _speeds(
    poses: np.ndarray, fps: float, motion_span_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    translation = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) * 1000.0 * fps
    relative = np.einsum(
        "nji,njk->nik", poses[:-1, :3, :3], poses[1:, :3, :3], optimize=True
    )
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation = np.degrees(np.arccos(cosine)) * fps
    half_span = motion_span_frames // 2
    indices = np.arange(len(poses))
    begin = np.maximum(0, indices - half_span)
    end = np.minimum(len(poses) - 1, indices + half_span)
    elapsed = (end - begin) / fps
    coarse_translation = (
        np.linalg.norm(poses[end, :3, 3] - poses[begin, :3, 3], axis=1) * 1000.0 / elapsed
    )
    coarse_relative = np.einsum(
        "nji,njk->nik",
        poses[begin, :3, :3],
        poses[end, :3, :3],
        optimize=True,
    )
    coarse_cosine = np.clip(
        (np.trace(coarse_relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0
    )
    coarse_rotation = np.degrees(np.arccos(coarse_cosine)) / elapsed
    return translation, rotation, coarse_translation, coarse_rotation


def _eligible_mask(
    moving_frames: np.ndarray,
    hard_steps: np.ndarray,
    frames: int,
    radius: int,
    minimum_enabled_run_frames: int,
) -> np.ndarray:
    difference = np.zeros(frames + 1, dtype=np.int32)
    for frame in np.flatnonzero(moving_frames):
        begin = max(0, int(frame) - radius)
        end = min(frames - 1, int(frame) + radius)
        difference[begin] += 1
        difference[end + 1] -= 1
    for step in np.flatnonzero(hard_steps):
        begin = max(0, int(step) - radius + 1)
        end = min(frames - 1, int(step) + radius)
        difference[begin] += 1
        difference[end + 1] -= 1
    enabled = np.cumsum(difference[:-1]) == 0
    transitions = np.diff(np.pad(enabled.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    for begin, end in zip(starts, ends):
        if end - begin < minimum_enabled_run_frames:
            enabled[begin:end] = False
    return enabled


def _summary(values: np.ndarray) -> dict[str, float]:
    p = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
    return {
        key: float(value)
        for key, value in zip(("min", "p10", "p25", "p50", "p75", "p90", "max"), p)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("final-npz", "latest-records"), default="final-npz")
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--npz-name", default="object_6d_pose_v2.npz")
    parser.add_argument("--latest-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--motion-span-frames", type=int, default=11)
    parser.add_argument("--min-run-frames", type=int, default=15)
    parser.add_argument("--hard-translation-mm-s", type=float, default=300.0)
    parser.add_argument("--hard-rotation-deg-s", type=float, default=300.0)
    parser.add_argument(
        "--threshold", nargs=2, type=float, action="append", default=None,
        metavar=("TRANSLATION_MM_S", "ROTATION_DEG_S"),
        help="Repeatable. Defaults: 15/7.5, 30/15, and 60/30.",
    )
    parser.add_argument("--top-cases", type=int, default=12)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.window_size < 1 or args.window_size % 2 != 1:
        raise ValueError("FPS must be positive and window size a positive odd integer")
    if args.motion_span_frames < 3 or args.motion_span_frames % 2 != 1:
        raise ValueError("Motion span must be odd and at least 3")
    if args.min_run_frames < 1:
        raise ValueError("Minimum enabled run must be positive")
    if min(args.hard_translation_mm_s, args.hard_rotation_deg_s) <= 0:
        raise ValueError("Hard thresholds must be positive")
    thresholds = args.threshold or [[15.0, 7.5], [30.0, 15.0], [60.0, 30.0]]
    if any(min(pair) <= 0 for pair in thresholds):
        raise ValueError("Thresholds must be positive")
    trajectories: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    manifest_path: Path | None = None
    campaign: str | None = None
    if args.source == "final-npz":
        target_root = Path(args.target_root).expanduser().resolve()
        if not target_root.is_dir():
            raise FileNotFoundError(target_root)
        inputs = [
            (f"{path.parent.parent.name}/{path.parent.name}", path, _load_npz_poses)
            for path in sorted(target_root.glob(f"*/*/{args.npz_name}"))
        ]
        campaign = target_root.name
    else:
        manifest_path = Path(args.latest_manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("latest manifest has no episode list")
        inputs = [
            (
                f"{entry.get('object_name')}/{entry.get('episode')}",
                Path(str(entry.get("record", ""))).expanduser(),
                _load_poses,
            )
            for entry in episodes
            if isinstance(entry, dict) and entry.get("latest_status") == "completed"
        ]
        campaign = manifest.get("campaign")

    for name, path, loader in inputs:
        try:
            poses = loader(path)
            translation, rotation, coarse_translation, coarse_rotation = _speeds(
                poses, args.fps, args.motion_span_frames
            )
        except Exception as exc:
            skipped.append({"case": name, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        trajectories.append({
            "case": name,
            "record": str(path),
            "frames": len(poses),
            "translation_speed": translation,
            "rotation_speed": rotation,
            "coarse_translation_speed": coarse_translation,
            "coarse_rotation_speed": coarse_rotation,
        })
    if not trajectories:
        raise RuntimeError("No usable trajectories found")

    radius = args.window_size // 2
    reports: list[dict[str, Any]] = []
    for translation_threshold, rotation_threshold in thresholds:
        cases: list[dict[str, Any]] = []
        total_frames = total_translation_enabled = total_rotation_enabled = total_both_enabled = 0
        translation_motion = rotation_motion = either_motion = total_motion_frames = 0
        hard_crossings = total_steps = 0
        for trajectory in trajectories:
            translation_fast = trajectory["coarse_translation_speed"] > translation_threshold
            rotation_fast = trajectory["coarse_rotation_speed"] > rotation_threshold
            translation_hard = trajectory["translation_speed"] > args.hard_translation_mm_s
            rotation_hard = trajectory["rotation_speed"] > args.hard_rotation_deg_s
            translation_enabled = _eligible_mask(
                translation_fast,
                translation_hard,
                trajectory["frames"],
                radius,
                args.min_run_frames,
            )
            rotation_enabled = _eligible_mask(
                rotation_fast,
                rotation_hard,
                trajectory["frames"],
                radius,
                args.min_run_frames,
            )
            both_enabled = translation_enabled & rotation_enabled
            fraction = float(both_enabled.mean())
            endpoint_frames = min(30, trajectory["frames"])
            cases.append({
                "case": trajectory["case"],
                "frames": trajectory["frames"],
                "enabled_frames": int(both_enabled.sum()),
                "enabled_fraction": fraction,
                "translation_enabled_fraction": float(translation_enabled.mean()),
                "rotation_enabled_fraction": float(rotation_enabled.mean()),
                "both_enabled_fraction": fraction,
                "start_30_enabled_fraction": float(both_enabled[:endpoint_frames].mean()),
                "end_30_enabled_fraction": float(both_enabled[-endpoint_frames:].mean()),
                "start_30_translation_enabled_fraction": float(
                    translation_enabled[:endpoint_frames].mean()
                ),
                "end_30_translation_enabled_fraction": float(
                    translation_enabled[-endpoint_frames:].mean()
                ),
                "start_30_rotation_enabled_fraction": float(
                    rotation_enabled[:endpoint_frames].mean()
                ),
                "end_30_rotation_enabled_fraction": float(
                    rotation_enabled[-endpoint_frames:].mean()
                ),
                "translation_motion_fraction": float(translation_fast.mean()),
                "rotation_motion_fraction": float(rotation_fast.mean()),
                "either_motion_fraction": float((translation_fast | rotation_fast).mean()),
                "hard_step_fraction": float((translation_hard | rotation_hard).mean()),
                "translation_hard_step_fraction": float(translation_hard.mean()),
                "rotation_hard_step_fraction": float(rotation_hard.mean()),
            })
            total_frames += trajectory["frames"]
            total_translation_enabled += int(translation_enabled.sum())
            total_rotation_enabled += int(rotation_enabled.sum())
            total_both_enabled += int(both_enabled.sum())
            total_motion_frames += len(translation_fast)
            translation_motion += int(translation_fast.sum())
            rotation_motion += int(rotation_fast.sum())
            either_motion += int((translation_fast | rotation_fast).sum())
            total_steps += len(translation_hard)
            hard_crossings += int((translation_hard | rotation_hard).sum())
        fractions = np.asarray([case["enabled_fraction"] for case in cases])
        by_enabled = sorted(cases, key=lambda case: (case["enabled_fraction"], case["case"]))
        report = {
            "translation_threshold_mm_s": translation_threshold,
            "rotation_threshold_deg_s": rotation_threshold,
            "aggregate_enabled_fraction": total_both_enabled / total_frames,
            "aggregate_translation_enabled_fraction": total_translation_enabled / total_frames,
            "aggregate_rotation_enabled_fraction": total_rotation_enabled / total_frames,
            "aggregate_both_enabled_fraction": total_both_enabled / total_frames,
            "case_enabled_fraction": _summary(fractions),
            "cases_with_any_enabled": sum(case["enabled_frames"] > 0 for case in cases),
            "cases_at_least_half_enabled": sum(case["enabled_fraction"] >= 0.5 for case in cases),
            "cases_at_least_90pct_enabled": sum(case["enabled_fraction"] >= 0.9 for case in cases),
            "translation_motion_fraction": translation_motion / total_motion_frames,
            "rotation_motion_fraction": rotation_motion / total_motion_frames,
            "either_motion_fraction": either_motion / total_motion_frames,
            "hard_step_fraction": hard_crossings / total_steps,
            "least_enabled_cases": by_enabled[:args.top_cases],
            "most_enabled_cases": list(reversed(by_enabled[-args.top_cases:])),
            "cases": cases,
        }
        reports.append(report)
        print(
            f"[gate] {translation_threshold:g} mm/s, {rotation_threshold:g} deg/s: "
            f"translation={report['aggregate_translation_enabled_fraction']:.1%}; "
            f"rotation={report['aggregate_rotation_enabled_fraction']:.1%}; "
            f"both={report['aggregate_both_enabled_fraction']:.1%}; "
            f"case_median={report['case_enabled_fraction']['p50']:.1%}; "
            f"cases>=50%={report['cases_at_least_half_enabled']}/{len(cases)}; "
            f"motion={report['either_motion_fraction']:.1%}; "
            f"hard_step={report['hard_step_fraction']:.1%}",
            flush=True,
        )

    result = {
        "source": args.source,
        "manifest": str(manifest_path) if manifest_path else None,
        "target_root": str(Path(args.target_root).expanduser().resolve()),
        "campaign": campaign,
        "fps": args.fps,
        "window_size": args.window_size,
        "motion_span_frames": args.motion_span_frames,
        "minimum_enabled_run_frames": args.min_run_frames,
        "hard_translation_threshold_mm_s": args.hard_translation_mm_s,
        "hard_rotation_threshold_deg_s": args.hard_rotation_deg_s,
        "trajectories": len(trajectories),
        "frames": int(sum(item["frames"] for item in trajectories)),
        "skipped": skipped,
        "thresholds": reports,
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to replace {output}")
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[gate] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
