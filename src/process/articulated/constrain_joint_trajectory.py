#!/usr/bin/env python3
"""Constrain a tracked one-dimensional joint trajectory by sparse measurements.

GoTrack can return a coherent but wrong coordinate for a moving part.  Applying
an external measurement only at the measured frame resets the tracker, but does not
stop it from jumping back to the same visual alias one frame later.  This post-pass
therefore treats the sparse measurements as hard constraints on the *whole* scalar
trajectory while leaving every 6-D body pose untouched.

The default objective is the minimum-velocity trajectory through all stereo anchors::

    minimize  sum_t (d[t] - d[t-1])**2
    subject to d[k] = stereo[k] for every measured frame k

Between two adjacent anchors this is exactly linear interpolation.  Expressing it as
a constrained least-squares problem makes the intended extension points explicit:
an acceleration prior and weak, clipped GoTrack delta measurements can be enabled,
but both default to zero because drawer/2 demonstrates that the flow can be
confidently and coherently wrong.  Frames outside the measured span hold the nearest
measurement rather than extrapolating an unverified motion.

``--constraint-mode soft`` is intended for a denser adaptive pass.  Stereo then
becomes a robust measurement factor instead of an equality.  A second-difference
prior suppresses measurement-to-measurement jitter.  For mechanisms known to hit an
upper travel stop, ``--upper-plateau-tolerance`` adds a stop factor to the contiguous
near-maximum measurements around the observed peak; this is opt-in because smooth
motion alone cannot distinguish "still opening" from a biased measurement just
before a stop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_anchors(path: Path, object_name: str) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload.get(object_name)
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{path}: no non-empty anchor table for {object_name!r}")
    anchors = {int(frame): float(value) for frame, value in table.items()}
    if not all(np.isfinite(value) for value in anchors.values()):
        raise ValueError(f"{path}: anchor values must be finite")
    return anchors


def solve_trajectory(
    frames: np.ndarray,
    raw: np.ndarray,
    anchors: dict[int, float],
    *,
    acceleration_weight: float = 0.0,
    flow_delta_weight: float = 0.0,
    max_flow_step: float | None = None,
) -> np.ndarray:
    """Return one joint value per frame, satisfying every in-span anchor exactly."""
    frames = np.asarray(frames, dtype=np.int64)
    raw = np.asarray(raw, dtype=np.float64)
    if frames.ndim != 1 or raw.shape != frames.shape or frames.size == 0:
        raise ValueError("frames and raw must be non-empty one-dimensional arrays")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("frames must be strictly increasing")
    if acceleration_weight < 0.0 or flow_delta_weight < 0.0:
        raise ValueError("weights must be non-negative")
    if max_flow_step is not None and max_flow_step <= 0.0:
        raise ValueError("max_flow_step must be positive")

    frame_to_slot = {int(frame): slot for slot, frame in enumerate(frames)}
    used = {frame_to_slot[frame]: value for frame, value in anchors.items()
            if frame in frame_to_slot}
    if not used:
        raise ValueError("none of the stereo anchor frames occurs in the trajectory")

    n = frames.size
    rows: list[np.ndarray] = []
    targets: list[float] = []

    # A first-difference prior is a temporal factor on every adjacent pair.  Divide
    # by the frame gap so missing records do not silently create a stronger factor.
    for i in range(1, n):
        dt = float(frames[i] - frames[i - 1])
        row = np.zeros(n, dtype=np.float64)
        row[i - 1], row[i] = -1.0 / dt, 1.0 / dt
        rows.append(row)
        targets.append(0.0)

    if acceleration_weight > 0.0 and n >= 3:
        scale = float(np.sqrt(acceleration_weight))
        for i in range(1, n - 1):
            dt0 = float(frames[i] - frames[i - 1])
            dt1 = float(frames[i + 1] - frames[i])
            row = np.zeros(n, dtype=np.float64)
            # Difference between the two neighbouring per-frame velocities.
            row[i - 1] = scale / dt0
            row[i] = -scale * (1.0 / dt0 + 1.0 / dt1)
            row[i + 1] = scale / dt1
            rows.append(row)
            targets.append(0.0)

    if flow_delta_weight > 0.0:
        scale = float(np.sqrt(flow_delta_weight))
        for i in range(1, n):
            if not (np.isfinite(raw[i - 1]) and np.isfinite(raw[i])):
                continue
            delta = float(raw[i] - raw[i - 1])
            if max_flow_step is not None:
                dt = float(frames[i] - frames[i - 1])
                delta = float(np.clip(delta, -max_flow_step * dt, max_flow_step * dt))
            row = np.zeros(n, dtype=np.float64)
            row[i - 1], row[i] = -scale, scale
            rows.append(row)
            targets.append(scale * delta)

    matrix = np.stack(rows)
    target = np.asarray(targets, dtype=np.float64)
    hessian = matrix.T @ matrix
    gradient = matrix.T @ target

    slots = np.asarray(sorted(used), dtype=np.int64)
    values = np.asarray([used[slot] for slot in slots], dtype=np.float64)
    constraint = np.zeros((slots.size, n), dtype=np.float64)
    constraint[np.arange(slots.size), slots] = 1.0

    # Exact equality constraints avoid a magic "very large" anchor weight and make
    # the output independent of whether joint values happen to use metres or radians.
    kkt = np.block([
        [hessian, constraint.T],
        [constraint, np.zeros((slots.size, slots.size), dtype=np.float64)],
    ])
    rhs = np.concatenate([gradient, values])
    solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
    result = solution[:n]

    # There is deliberately no unverified extrapolation beyond the measured range.
    first, last = int(slots.min()), int(slots.max())
    result[:first] = used[first]
    result[last + 1:] = used[last]
    result[slots] = values  # Preserve exact decimal anchor values in serialized JSON.
    return result


def solve_soft_trajectory(
    frames: np.ndarray,
    anchors: dict[int, float],
    *,
    measurement_weight: float = 1.0,
    acceleration_weight: float = 30.0,
    huber_delta: float = 0.01,
    upper_plateau_tolerance: float | None = None,
    plateau_weight: float = 30.0,
    joint_min: float | None = None,
    joint_max: float | None = None,
    irls_iterations: int = 8,
) -> tuple[np.ndarray, dict]:
    """Robust temporal fit where stereo values are measurements, not equalities."""
    frames = np.asarray(frames, dtype=np.int64)
    if frames.ndim != 1 or frames.size == 0 or np.any(np.diff(frames) <= 0):
        raise ValueError("frames must be a non-empty, strictly increasing vector")
    if measurement_weight <= 0.0 or acceleration_weight < 0.0:
        raise ValueError("measurement weight must be positive and acceleration non-negative")
    if huber_delta <= 0.0 or plateau_weight < 0.0 or irls_iterations < 1:
        raise ValueError("invalid robust/plateau solver parameters")
    if upper_plateau_tolerance is not None and upper_plateau_tolerance <= 0.0:
        raise ValueError("upper plateau tolerance must be positive")
    if joint_min is not None and joint_max is not None and joint_min > joint_max:
        raise ValueError("joint_min exceeds joint_max")

    frame_to_slot = {int(frame): slot for slot, frame in enumerate(frames)}
    used_frames = sorted(frame for frame in anchors if frame in frame_to_slot)
    if len(used_frames) < 2:
        raise ValueError("soft trajectory fitting needs at least two in-span anchors")
    slots = np.asarray([frame_to_slot[frame] for frame in used_frames], dtype=np.int64)
    values = np.asarray([anchors[frame] for frame in used_frames], dtype=np.float64)
    n = frames.size

    measurement = np.zeros((len(slots), n), dtype=np.float64)
    measurement[np.arange(len(slots)), slots] = 1.0
    acceleration = np.zeros((max(0, n - 2), n), dtype=np.float64)
    for i in range(1, n - 1):
        dt0 = float(frames[i] - frames[i - 1])
        dt1 = float(frames[i + 1] - frames[i])
        acceleration[i - 1, i - 1] = 1.0 / dt0
        acceleration[i - 1, i] = -(1.0 / dt0 + 1.0 / dt1)
        acceleration[i - 1, i + 1] = 1.0 / dt1

    plateau_frames: list[int] = []
    plateau_slots = np.empty(0, dtype=np.int64)
    plateau_target = None
    if upper_plateau_tolerance is not None:
        peak_index = int(np.argmax(values))
        plateau_target = float(values[peak_index])
        left = right = peak_index
        while left > 0 and values[left - 1] >= plateau_target - upper_plateau_tolerance:
            left -= 1
        while right + 1 < len(values) and values[right + 1] >= plateau_target - upper_plateau_tolerance:
            right += 1
        plateau_frames = used_frames[left:right + 1]
        plateau_slots = slots[left:right + 1]

    robust_weights = np.ones(len(slots), dtype=np.float64)
    result = np.interp(frames, np.asarray(used_frames), values)
    ridge = 1e-12 * np.eye(n)
    for _ in range(irls_iterations):
        weighted_measurement = measurement * np.sqrt(
            measurement_weight * robust_weights)[:, None]
        hessian = weighted_measurement.T @ weighted_measurement + ridge
        gradient = weighted_measurement.T @ (
            np.sqrt(measurement_weight * robust_weights) * values)
        if acceleration.size and acceleration_weight > 0.0:
            hessian += acceleration_weight * (acceleration.T @ acceleration)
        if plateau_slots.size and plateau_weight > 0.0:
            plateau = np.zeros((len(plateau_slots), n), dtype=np.float64)
            plateau[np.arange(len(plateau_slots)), plateau_slots] = 1.0
            hessian += plateau_weight * (plateau.T @ plateau)
            gradient += plateau_weight * plateau.T @ np.full(
                len(plateau_slots), plateau_target)
        result = np.linalg.solve(hessian, gradient)
        residual = result[slots] - values
        robust_weights = np.minimum(1.0, huber_delta / np.maximum(np.abs(residual), 1e-12))

    # Do not invent motion outside the measured interval. Bounds are a safety guard,
    # not another observation; clipping is normally inactive for a well-posed fit.
    result[:slots[0]] = result[slots[0]]
    result[slots[-1] + 1:] = result[slots[-1]]
    if joint_min is not None:
        result = np.maximum(result, joint_min)
    if joint_max is not None:
        result = np.minimum(result, joint_max)
    residual = result[slots] - values
    return result, {
        "plateau_frames": plateau_frames,
        "plateau_target": plateau_target,
        "mean_abs_measurement_residual": float(np.mean(np.abs(residual))),
        "max_abs_measurement_residual": float(np.max(np.abs(residual))),
        "num_huber_downweighted": int(np.count_nonzero(robust_weights < 1.0)),
    }


def constrain_records(
    records: list[dict],
    anchors: dict[int, float],
    *,
    constraint_mode: str = "hard",
    outside_mode: str = "hold",
    **solver_kwargs: float | None,
) -> tuple[list[dict], dict]:
    indexed = sorted(enumerate(records), key=lambda item: int(item[1]["frame_index"]))
    usable = [(original, record) for original, record in indexed
              if record.get("status") == "ok" and record.get("joint_value") is not None]
    if not usable:
        raise ValueError("trajectory contains no solved joint records")
    joint_types = {record.get("joint_type", "revolute") for _, record in usable}
    if len(joint_types) != 1 or not joint_types <= {"prismatic", "revolute"}:
        raise ValueError(f"records contain inconsistent/unsupported joint types: {sorted(joint_types)}")
    joint_type = next(iter(joint_types))

    frames = np.asarray([int(record["frame_index"]) for _, record in usable])
    raw = np.asarray([
        float(record.get("joint_value_before_temporal_constraint",
                         record.get("joint_value_tracked", record["joint_value"])))
        for _, record in usable
    ])
    if constraint_mode == "hard":
        constrained = solve_trajectory(frames, raw, anchors, **solver_kwargs)
        solver_report: dict = {}
    elif constraint_mode == "soft":
        constrained, solver_report = solve_soft_trajectory(
            frames, anchors, **solver_kwargs)
    else:
        raise ValueError(f"unknown constraint mode {constraint_mode!r}")
    if outside_mode not in {"hold", "raw"}:
        raise ValueError(f"unknown outside mode {outside_mode!r}")
    if outside_mode == "raw":
        used_frames = sorted(set(map(int, frames)) & set(anchors))
        if not used_frames:
            raise ValueError("none of the anchor frames occurs in the trajectory")
        inside = (frames >= used_frames[0]) & (frames <= used_frames[-1])
        constrained = np.where(inside, constrained, raw)

    output = [dict(record) for record in records]
    corrections = []
    anchor_frames = set(anchors)
    for (original, record), value, raw_value in zip(usable, constrained, raw):
        updated = output[original]
        updated["joint_value_before_temporal_constraint"] = float(raw_value)
        updated["joint_value"] = float(value)
        updated["joint_temporal_constraint_applied"] = True
        updated["joint_temporal_constraint_source"] = "sparse_depth"
        updated["joint_temporal_constraint_mode"] = constraint_mode
        if int(record["frame_index"]) in anchor_frames:
            updated["joint_anchor_applied"] = True
        corrections.append(float(value - raw_value))

    used_frames = sorted(set(map(int, frames)) & anchor_frames)
    report = {
        "frames": int(frames.size),
        "anchor_frames_used": used_frames,
        "num_anchor_frames_used": len(used_frames),
        "first_anchor_frame": used_frames[0],
        "last_anchor_frame": used_frames[-1],
        "max_abs_correction": float(np.max(np.abs(corrections))),
        "mean_abs_correction": float(np.mean(np.abs(corrections))),
        "max_anchor_error": float(max(
            abs(output[original]["joint_value"] - anchors[int(record["frame_index"])])
            for original, record in usable if int(record["frame_index"]) in anchor_frames)),
        "constraint_mode": constraint_mode,
        "joint_type": joint_type,
        "outside_mode": outside_mode,
        **solver_report,
    }
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="GoTrack output containing <object>/world_pose_records.json")
    parser.add_argument("--object", required=True)
    parser.add_argument("--joint-anchors", type=Path, required=True)
    parser.add_argument("--constraint-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--outside-mode", choices=("hold", "raw"), default="hold",
                        help="hold keeps the nearest anchor outside its measured span "
                             "(drawer default); raw preserves the source trajectory "
                             "outside a targeted correction interval.")
    parser.add_argument("--measurement-weight", type=float, default=1.0)
    parser.add_argument("--acceleration-weight", type=float, default=None,
                        help="Second-difference weight. Defaults to 0 in hard mode and "
                             "30 in soft mode.")
    parser.add_argument("--huber-delta", type=float, default=0.01)
    parser.add_argument("--upper-plateau-tolerance", type=float, default=None,
                        help="Opt-in travel-stop factor: contiguous measurements this "
                             "close to the observed maximum share its plateau target.")
    parser.add_argument("--plateau-weight", type=float, default=30.0)
    parser.add_argument("--joint-min", type=float, default=None)
    parser.add_argument("--joint-max", type=float, default=None)
    parser.add_argument("--flow-delta-weight", type=float, default=0.0,
                        help="Weakly preserve GoTrack per-frame deltas. Default 0 because "
                             "coherent flow drift is the failure this pass corrects.")
    parser.add_argument("--max-flow-step", type=float, default=None,
                        help="Clip a GoTrack delta to this magnitude per frame, in the "
                             "joint's native unit. Only used with --flow-delta-weight.")
    args = parser.parse_args()

    record_path = args.run_dir / args.object / "world_pose_records.json"
    records = json.loads(record_path.read_text(encoding="utf-8"))
    anchors = load_anchors(args.joint_anchors, args.object)
    acceleration_weight = (30.0 if args.constraint_mode == "soft" else 0.0)
    if args.acceleration_weight is not None:
        acceleration_weight = args.acceleration_weight
    if args.constraint_mode == "soft":
        solver_kwargs = {
            "measurement_weight": args.measurement_weight,
            "acceleration_weight": acceleration_weight,
            "huber_delta": args.huber_delta,
            "upper_plateau_tolerance": args.upper_plateau_tolerance,
            "plateau_weight": args.plateau_weight,
            "joint_min": args.joint_min,
            "joint_max": args.joint_max,
        }
    else:
        solver_kwargs = {
            "acceleration_weight": acceleration_weight,
            "flow_delta_weight": args.flow_delta_weight,
            "max_flow_step": args.max_flow_step,
        }
    output, report = constrain_records(
        records, anchors, constraint_mode=args.constraint_mode,
        outside_mode=args.outside_mode, **solver_kwargs)
    record_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    report.update({
        "object": args.object,
        "record_path": str(record_path),
        "joint_anchors": str(args.joint_anchors),
        "constraint_mode": args.constraint_mode,
        "outside_mode": args.outside_mode,
        "measurement_weight": args.measurement_weight,
        "acceleration_weight": acceleration_weight,
        "huber_delta": args.huber_delta,
        "upper_plateau_tolerance": args.upper_plateau_tolerance,
        "plateau_weight": args.plateau_weight,
        "flow_delta_weight": args.flow_delta_weight,
        "max_flow_step": args.max_flow_step,
        "body_pose_modified": False,
    })
    report_path = args.run_dir / "joint_trajectory_constraints.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"constrained {report['frames']} frames through "
          f"{report['num_anchor_frames_used']} stereo anchors")
    print(f"  mean |correction| {report['mean_abs_correction']:.6f}, "
          f"max {report['max_abs_correction']:.6f}, "
          f"measurement residual {report['max_anchor_error']:.3e}")
    print(f"wrote {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
