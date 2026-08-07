#!/usr/bin/env python3
"""Offline temporal filters for GoTrack world-pose records.

The source JSON is never modified. Three filters are available:

* ``gaussian``: a symmetric Gaussian translation/SO(3) local-mean filter.
* ``gated-gaussian``: Gaussian only when every step touching its window is slow.
* ``one-euro``: a causal speed-adaptive low-pass filter on translation and SO(3).

Missing poses and non-consecutive frame ranges split the input into independent
segments. The output keeps all redundant world-pose fields consistent and is
accompanied by a manifest with source identity, settings, and correction stats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _as_pose(value: object) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Expected a finite 4x4 pose, got {pose.shape}")
    return pose


def _segments(rows: list[dict[str, Any]]) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    previous_frame: int | None = None
    for row_index, row in enumerate(rows):
        pose = row.get("pose_world")
        try:
            frame = int(row["frame_index"])
            if pose is not None:
                _as_pose(pose)
        except (KeyError, TypeError, ValueError):
            pose = None
        if pose is None:
            if current:
                segments.append(current)
                current = []
            previous_frame = None
            continue
        if previous_frame is not None and frame != previous_frame + 1:
            segments.append(current)
            current = []
        current.append(row_index)
        previous_frame = frame
    if current:
        segments.append(current)
    return segments


def _smooth_segment(poses: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    translations = poses[:, :3, 3]
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    output = poses.copy()
    for index in range(len(poses)):
        begin = max(0, index - radius)
        end = min(len(poses), index + radius + 1)
        offsets = np.arange(begin, end, dtype=np.float64) - index
        weights = np.exp(-0.5 * np.square(offsets / sigma))
        weights /= weights.sum()
        output[index, :3, 3] = np.sum(translations[begin:end] * weights[:, None], axis=0)

        reference = rotations[index]
        tangent = (reference.inv() * rotations[begin:end]).as_rotvec()
        output[index, :3, :3] = (
            reference * Rotation.from_rotvec(np.sum(tangent * weights[:, None], axis=0))
        ).as_matrix()
    output[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return output


def _gated_gaussian_segment(
    poses: np.ndarray,
    radius: int,
    sigma: float,
    *,
    fps: float,
    translation_threshold_mm_s: float,
    rotation_threshold_deg_s: float,
    motion_span_frames: int,
    minimum_enabled_run_frames: int,
    hard_translation_threshold_mm_s: float,
    hard_rotation_threshold_deg_s: float,
    transition_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently smooth stationary translation and rotation components."""
    gaussian = _smooth_segment(poses, radius, sigma)
    enabled = np.ones(len(poses), dtype=bool)
    if len(poses) < 2:
        return gaussian, enabled, enabled.copy()
    translation_speed = (
        np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) * 1000.0 * fps
    )
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    rotation_speed = np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude()) * fps
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    half_span = motion_span_frames // 2
    coarse_translation_speed = np.zeros(len(poses), dtype=np.float64)
    coarse_rotation_speed = np.zeros(len(poses), dtype=np.float64)
    for index in range(len(poses)):
        begin = max(0, index - half_span)
        end = min(len(poses) - 1, index + half_span)
        elapsed = (end - begin) / fps
        if elapsed <= 0.0:
            continue
        coarse_translation_speed[index] = (
            np.linalg.norm(poses[end, :3, 3] - poses[begin, :3, 3]) * 1000.0 / elapsed
        )
        coarse_rotation_speed[index] = (
            np.degrees((rotations[begin].inv() * rotations[end]).magnitude()) / elapsed
        )
    def component_gate(moving: np.ndarray, hard: np.ndarray) -> np.ndarray:
        difference = np.zeros(len(poses) + 1, dtype=np.int32)
        # If a frame is moving, every Gaussian window touching it remains raw.
        for frame in np.flatnonzero(moving):
            begin = max(0, int(frame) - radius)
            end = min(len(poses) - 1, int(frame) + radius)
            difference[begin] += 1
            difference[end + 1] -= 1
        # A hard step j connects poses j and j+1 and guards against a jump being
        # averaged even when the longer-span displacement happens to cancel out.
        for step in np.flatnonzero(hard):
            begin = max(0, int(step) - radius + 1)
            end = min(len(poses) - 1, int(step) + radius)
            difference[begin] += 1
            difference[end + 1] -= 1
        component_enabled = np.cumsum(difference[:-1]) == 0
        transitions = np.diff(np.pad(component_enabled.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        for begin, end in zip(starts, ends):
            if end - begin < minimum_enabled_run_frames:
                component_enabled[begin:end] = False
        return component_enabled

    translation_enabled = component_gate(
        coarse_translation_speed > translation_threshold_mm_s,
        translation_speed > hard_translation_threshold_mm_s,
    )
    rotation_enabled = component_gate(
        coarse_rotation_speed > rotation_threshold_deg_s,
        rotation_speed > hard_rotation_threshold_deg_s,
    )

    def blend_weights(component_enabled: np.ndarray) -> np.ndarray:
        """Ramp correction only inside enabled runs adjacent to raw frames."""
        weights = component_enabled.astype(np.float64)
        if transition_frames == 0:
            return weights
        transitions = np.diff(np.pad(component_enabled.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        ramp = np.arange(1, transition_frames + 1, dtype=np.float64) / (
            transition_frames + 1
        )
        for begin, end in zip(starts, ends):
            run_length = end - begin
            count = min(transition_frames, run_length)
            if begin > 0:
                weights[begin:begin + count] = np.minimum(
                    weights[begin:begin + count], ramp[:count]
                )
            if end < len(component_enabled):
                weights[end - count:end] = np.minimum(
                    weights[end - count:end], ramp[:count][::-1]
                )
        return weights

    translation_weights = blend_weights(translation_enabled)
    rotation_weights = blend_weights(rotation_enabled)
    output = poses.copy()
    output[:, :3, 3] += translation_weights[:, None] * (
        gaussian[:, :3, 3] - poses[:, :3, 3]
    )
    raw_rotations = Rotation.from_matrix(poses[:, :3, :3])
    gaussian_rotations = Rotation.from_matrix(gaussian[:, :3, :3])
    correction = (raw_rotations.inv() * gaussian_rotations).as_rotvec()
    output[:, :3, :3] = (
        raw_rotations * Rotation.from_rotvec(rotation_weights[:, None] * correction)
    ).as_matrix()
    return output, translation_enabled, rotation_enabled


def _lowpass_alpha(cutoff_hz: float, dt: float) -> float:
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive and finite, got {cutoff_hz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt}")
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


def _one_euro_segment(
    poses: np.ndarray,
    times: np.ndarray,
    *,
    translation_min_cutoff_hz: float,
    translation_beta: float,
    rotation_min_cutoff_hz: float,
    rotation_beta: float,
    derivative_cutoff_hz: float,
) -> np.ndarray:
    """Apply a vector-isotropic One Euro filter on R^3 x SO(3).

    Translation and angular-speed magnitudes control separate adaptive cutoff
    frequencies. Rotation interpolation follows the SO(3) geodesic, avoiding
    Euler-angle and rotation-matrix component averaging.
    """
    output = poses.copy()
    if len(poses) < 2:
        return output

    filtered_translation = poses[0, :3, 3].copy()
    previous_raw_translation = filtered_translation.copy()
    filtered_rotation = Rotation.from_matrix(poses[0, :3, :3])
    previous_raw_rotation = filtered_rotation
    filtered_linear_velocity = np.zeros(3, dtype=np.float64)
    filtered_angular_speed = 0.0

    for index in range(1, len(poses)):
        dt = float(times[index] - times[index - 1])
        raw_translation = poses[index, :3, 3]
        raw_rotation = Rotation.from_matrix(poses[index, :3, :3])

        derivative_alpha = _lowpass_alpha(derivative_cutoff_hz, dt)
        raw_linear_velocity = (raw_translation - previous_raw_translation) / dt
        filtered_linear_velocity = (
            derivative_alpha * raw_linear_velocity
            + (1.0 - derivative_alpha) * filtered_linear_velocity
        )
        translation_cutoff = (
            translation_min_cutoff_hz
            + translation_beta * float(np.linalg.norm(filtered_linear_velocity))
        )
        translation_alpha = _lowpass_alpha(translation_cutoff, dt)
        filtered_translation = (
            translation_alpha * raw_translation
            + (1.0 - translation_alpha) * filtered_translation
        )

        raw_angular_speed = float(
            np.linalg.norm((previous_raw_rotation.inv() * raw_rotation).as_rotvec()) / dt
        )
        filtered_angular_speed = (
            derivative_alpha * raw_angular_speed
            + (1.0 - derivative_alpha) * filtered_angular_speed
        )
        rotation_cutoff = (
            rotation_min_cutoff_hz
            + rotation_beta * filtered_angular_speed
        )
        rotation_alpha = _lowpass_alpha(rotation_cutoff, dt)
        innovation = (filtered_rotation.inv() * raw_rotation).as_rotvec()
        filtered_rotation = filtered_rotation * Rotation.from_rotvec(rotation_alpha * innovation)

        output[index, :3, 3] = filtered_translation
        output[index, :3, :3] = filtered_rotation.as_matrix()
        previous_raw_translation = raw_translation
        previous_raw_rotation = raw_rotation

    output[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return output


def _segment_times(
    rows: list[dict[str, Any]], indices: list[int], time_source: str, fps: float
) -> tuple[np.ndarray, str]:
    frames = np.asarray([int(rows[index]["frame_index"]) for index in indices], dtype=np.float64)
    frame_times = (frames - frames[0]) / fps
    if time_source == "frame-index":
        return frame_times, "frame-index"

    try:
        record_times = np.asarray([float(rows[index]["time_sec"]) for index in indices])
        valid = bool(np.isfinite(record_times).all() and np.all(np.diff(record_times) > 0.0))
    except (KeyError, TypeError, ValueError):
        valid = False
        record_times = np.empty(0, dtype=np.float64)
    if valid:
        return record_times - record_times[0], "record-time"
    if time_source == "record-time":
        raise ValueError("--time-source record-time requires finite, strictly increasing time_sec")
    return frame_times, "frame-index-fallback"


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {key: 0.0 for key in ("p50", "p90", "p95", "p99", "max")}
    result = np.percentile(values, [50, 90, 95, 99, 100])
    return {key: float(value) for key, value in zip(("p50", "p90", "p95", "p99", "max"), result)}


def _metrics(originals: list[np.ndarray], smoothed_values: list[np.ndarray]) -> dict[str, Any]:
    original = np.concatenate(originals)
    smoothed = np.concatenate(smoothed_values)
    original_r = Rotation.from_matrix(original[:, :3, :3])
    smoothed_r = Rotation.from_matrix(smoothed[:, :3, :3])
    correction_t = np.linalg.norm(smoothed[:, :3, 3] - original[:, :3, 3], axis=1) * 1000.0
    correction_r = np.degrees((original_r.inv() * smoothed_r).magnitude())
    raw_step_t = np.concatenate([
        np.linalg.norm(np.diff(value[:, :3, 3], axis=0), axis=1) * 1000.0 for value in originals
    ])
    smooth_step_t = np.concatenate([
        np.linalg.norm(np.diff(value[:, :3, 3], axis=0), axis=1) * 1000.0 for value in smoothed_values
    ])
    raw_step_r = np.concatenate([
        np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude())
        for rotation in (Rotation.from_matrix(value[:, :3, :3]) for value in originals)
    ])
    smooth_step_r = np.concatenate([
        np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude())
        for rotation in (Rotation.from_matrix(value[:, :3, :3]) for value in smoothed_values)
    ])
    metrics = {
        "translation_correction_mm": _percentiles(correction_t),
        "rotation_correction_deg": _percentiles(correction_r),
        "raw_translation_step_mm": _percentiles(raw_step_t),
        "smoothed_translation_step_mm": _percentiles(smooth_step_t),
        "raw_rotation_step_deg": _percentiles(raw_step_r),
        "smoothed_rotation_step_deg": _percentiles(smooth_step_r),
    }
    metrics["translation_peak_step_preservation_ratio"] = float(
        smooth_step_t.max() / raw_step_t.max()
    ) if len(raw_step_t) and raw_step_t.max() > 0.0 else 1.0
    metrics["rotation_peak_step_preservation_ratio"] = float(
        smooth_step_r.max() / raw_step_r.max()
    ) if len(raw_step_r) and raw_step_r.max() > 0.0 else 1.0
    metrics["translation_path_length_ratio"] = float(
        smooth_step_t.sum() / raw_step_t.sum()
    ) if len(raw_step_t) and raw_step_t.sum() > 0.0 else 1.0
    metrics["rotation_path_length_ratio"] = float(
        smooth_step_r.sum() / raw_step_r.sum()
    ) if len(raw_step_r) and raw_step_r.sum() > 0.0 else 1.0
    return metrics


def _update_pose_fields(row: dict[str, Any], pose: np.ndarray) -> None:
    row["pose_world"] = pose.tolist()
    if "rotation_world" in row:
        row["rotation_world"] = pose[:3, :3].tolist()
    if "translation_world_m" in row:
        row["translation_world_m"] = pose[:3, 3].tolist()
    if "translation_world_mm" in row:
        row["translation_world_mm"] = (pose[:3, 3] * 1000.0).tolist()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source world_pose_records.json")
    parser.add_argument("--output", default=None,
                        help="Default: <input-stem>_smoothed.json beside the input")
    parser.add_argument("--manifest", default=None,
                        help="Default: <output-stem>.manifest.json")
    parser.add_argument(
        "--method", choices=("gaussian", "gated-gaussian", "one-euro"),
        default="gated-gaussian",
        help="Default: gated-gaussian.",
    )
    parser.add_argument("--window-size", type=int, default=7,
                        help="Gaussian methods: odd symmetric window size. Default: 7.")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Gaussian methods: sigma in frames. Default: 1.5.")
    parser.add_argument("--gate-translation-mm-s", type=float, default=15.0,
                        help="Gated Gaussian maximum translation speed. Default: 15 mm/s.")
    parser.add_argument("--gate-rotation-deg-s", type=float, default=7.5,
                        help="Gated Gaussian maximum angular speed. Default: 7.5 deg/s.")
    parser.add_argument("--gate-motion-span-frames", type=int, default=11,
                        help="Odd span used to estimate net motion. Default: 11 frames.")
    parser.add_argument("--gate-min-run-frames", type=int, default=15,
                        help="Minimum stationary run to enable smoothing. Default: 15 frames.")
    parser.add_argument("--gate-transition-frames", type=int, default=3,
                        help="Correction ramp inside each raw/smoothed boundary. Default: 3 frames.")
    parser.add_argument("--gate-hard-translation-mm-s", type=float, default=300.0,
                        help="Raw-step safety cutoff. Default: 300 mm/s (10 mm/frame at 30 FPS).")
    parser.add_argument("--gate-hard-rotation-deg-s", type=float, default=300.0,
                        help="Raw-step safety cutoff. Default: 300 deg/s (10 deg/frame at 30 FPS).")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Fallback FPS for One Euro timing. Default: 30.")
    parser.add_argument(
        "--time-source", choices=("auto", "record-time", "frame-index"), default="auto",
        help="One Euro timing. auto rejects missing/non-monotonic record timestamps.",
    )
    parser.add_argument("--translation-min-cutoff-hz", type=float, default=1.0)
    parser.add_argument("--translation-beta", type=float, default=200.0,
                        help="One Euro translation speed coefficient (speed in m/s).")
    parser.add_argument("--rotation-min-cutoff-hz", type=float, default=1.0)
    parser.add_argument("--rotation-beta", type=float, default=5.0,
                        help="One Euro rotation speed coefficient (speed in rad/s).")
    parser.add_argument("--derivative-cutoff-hz", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.window_size < 1 or args.window_size % 2 != 1 or args.sigma <= 0:
        raise ValueError("--window-size must be a positive odd integer and --sigma must be positive")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if min(
        args.gate_translation_mm_s,
        args.gate_rotation_deg_s,
        args.gate_hard_translation_mm_s,
        args.gate_hard_rotation_deg_s,
    ) <= 0.0:
        raise ValueError("Gated Gaussian speed thresholds must be positive")
    if args.gate_motion_span_frames < 3 or args.gate_motion_span_frames % 2 != 1:
        raise ValueError("--gate-motion-span-frames must be odd and at least 3")
    if args.gate_min_run_frames < 1:
        raise ValueError("--gate-min-run-frames must be positive")
    if args.gate_transition_frames < 0:
        raise ValueError("--gate-transition-frames must be non-negative")
    one_euro_values = (
        args.translation_min_cutoff_hz,
        args.rotation_min_cutoff_hz,
        args.derivative_cutoff_hz,
    )
    if any(value <= 0.0 for value in one_euro_values) or min(
        args.translation_beta, args.rotation_beta
    ) < 0.0:
        raise ValueError("One Euro cutoffs must be positive and beta values non-negative")
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = (Path(args.output).expanduser().resolve() if args.output else
              source.with_name(f"{source.stem}_smoothed.json"))
    manifest = (Path(args.manifest).expanduser().resolve() if args.manifest else
                output.with_name(f"{output.stem}.manifest.json"))
    for path in (output, manifest):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to replace {path}; pass --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected the input JSON root to be a list")
    output_rows = [dict(row) if isinstance(row, dict) else row for row in rows]
    segment_indices = _segments([row if isinstance(row, dict) else {} for row in rows])
    all_original: list[np.ndarray] = []
    all_smoothed: list[np.ndarray] = []
    timing_sources: list[str] = []
    translation_gate_masks: list[np.ndarray] = []
    rotation_gate_masks: list[np.ndarray] = []
    radius = args.window_size // 2
    for indices in segment_indices:
        original = np.asarray([_as_pose(rows[index]["pose_world"]) for index in indices])
        if args.method == "gaussian":
            smoothed = _smooth_segment(original, radius, args.sigma)
        elif args.method == "gated-gaussian":
            smoothed, translation_gate_mask, rotation_gate_mask = _gated_gaussian_segment(
                original,
                radius,
                args.sigma,
                fps=args.fps,
                translation_threshold_mm_s=args.gate_translation_mm_s,
                rotation_threshold_deg_s=args.gate_rotation_deg_s,
                motion_span_frames=args.gate_motion_span_frames,
                minimum_enabled_run_frames=args.gate_min_run_frames,
                hard_translation_threshold_mm_s=args.gate_hard_translation_mm_s,
                hard_rotation_threshold_deg_s=args.gate_hard_rotation_deg_s,
                transition_frames=args.gate_transition_frames,
            )
            translation_gate_masks.append(translation_gate_mask)
            rotation_gate_masks.append(rotation_gate_mask)
        else:
            times, used_time_source = _segment_times(rows, indices, args.time_source, args.fps)
            timing_sources.append(used_time_source)
            smoothed = _one_euro_segment(
                original,
                times,
                translation_min_cutoff_hz=args.translation_min_cutoff_hz,
                translation_beta=args.translation_beta,
                rotation_min_cutoff_hz=args.rotation_min_cutoff_hz,
                rotation_beta=args.rotation_beta,
                derivative_cutoff_hz=args.derivative_cutoff_hz,
            )
        for row_index, pose in zip(indices, smoothed):
            _update_pose_fields(output_rows[row_index], pose)
        all_original.append(original)
        all_smoothed.append(smoothed)
    if not all_original:
        raise ValueError("No valid pose_world records found")

    metadata = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(output),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "records_total": len(rows),
        "records_smoothed": int(sum(map(len, segment_indices))),
        "segments": len(segment_indices),
        "metrics": _metrics(all_original, all_smoothed),
    }
    if args.method in {"gaussian", "gated-gaussian"}:
        metadata["parameters"] = {
            "window_size": args.window_size,
            "sigma_frames": args.sigma,
        }
        if args.method == "gated-gaussian":
            translation_enabled = np.concatenate(translation_gate_masks)
            rotation_enabled = np.concatenate(rotation_gate_masks)
            both_enabled = translation_enabled & rotation_enabled

            def gate_summary(enabled: np.ndarray) -> dict[str, int | float]:
                transitions = np.diff(np.pad(enabled.astype(np.int8), (1, 1)))
                starts = np.flatnonzero(transitions == 1)
                ends = np.flatnonzero(transitions == -1)
                return {
                    "frames_enabled": int(enabled.sum()),
                    "frames_disabled": int(len(enabled) - enabled.sum()),
                    "enabled_fraction": float(enabled.mean()),
                    "enabled_runs": int(len(starts)),
                    "longest_enabled_run": int((ends - starts).max()) if len(starts) else 0,
                }

            metadata["parameters"].update({
                "fps": args.fps,
                "translation_threshold_mm_s": args.gate_translation_mm_s,
                "rotation_threshold_deg_s": args.gate_rotation_deg_s,
                "motion_span_frames": args.gate_motion_span_frames,
                "minimum_enabled_run_frames": args.gate_min_run_frames,
                "transition_frames": args.gate_transition_frames,
                "hard_translation_threshold_mm_s": args.gate_hard_translation_mm_s,
                "hard_rotation_threshold_deg_s": args.gate_hard_rotation_deg_s,
            })
            metadata["gate"] = {
                "translation": gate_summary(translation_enabled),
                "rotation": gate_summary(rotation_enabled),
                "both": gate_summary(both_enabled),
            }
    else:
        metadata["parameters"] = {
            "time_source_requested": args.time_source,
            "time_sources_used": sorted(set(timing_sources)),
            "fps_fallback": args.fps,
            "translation_min_cutoff_hz": args.translation_min_cutoff_hz,
            "translation_beta": args.translation_beta,
            "rotation_min_cutoff_hz": args.rotation_min_cutoff_hz,
            "rotation_beta": args.rotation_beta,
            "derivative_cutoff_hz": args.derivative_cutoff_hz,
        }
    _atomic_write(output, json.dumps(output_rows, indent=2) + "\n")
    _atomic_write(manifest, json.dumps(metadata, indent=2) + "\n")
    print(f"[smooth] input={source}")
    print(f"[smooth] output={output}")
    print(f"[smooth] manifest={manifest}")
    print(json.dumps(metadata["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
