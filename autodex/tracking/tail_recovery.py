"""Pure helpers for validating and merging a late GoTrack re-anchor.

The recovery pass starts near the end of a capture and tracks backwards to an
overlap with the original result.  These helpers deliberately do not launch
SAM3, FoundPose, or GoTrack; keeping the acceptance policy separate makes it
possible to test the safety-critical merge without a GPU.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def has_pose(record: Mapping[str, Any] | None) -> bool:
    return isinstance(record, Mapping) and record.get("pose_world") is not None


def records_by_frame(records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or "frame_index" not in record:
            continue
        indexed[int(record["frame_index"])] = dict(record)
    return indexed


def tail_gap(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Describe the terminal invalid run in one chronological record list."""
    indexed = records_by_frame(records)
    if not indexed:
        return {"last_frame": -1, "last_valid_frame": -1, "trailing_missing": 0}
    last_frame = max(indexed)
    valid_frames = [frame for frame, row in indexed.items() if has_pose(row)]
    last_valid = max(valid_frames) if valid_frames else -1
    return {
        "last_frame": last_frame,
        "last_valid_frame": last_valid,
        "trailing_missing": max(0, last_frame - last_valid),
    }


def descending_seed_frames(
    records: Sequence[Mapping[str, Any]],
    *,
    step: int,
    max_attempts: int,
    maximum_frame: int | None = None,
) -> list[int]:
    """Return late seed frames in end-to-front order, bounded by the gap."""
    if step < 1 or max_attempts < 1:
        raise ValueError("step and max_attempts must be positive")
    gap = tail_gap(records)
    if gap["last_valid_frame"] < 0:
        return []
    start = gap["last_frame"]
    if maximum_frame is not None:
        start = min(start, int(maximum_frame))
    if start <= gap["last_valid_frame"]:
        return []
    candidates: list[int] = []
    frame = start
    while frame > gap["last_valid_frame"] and len(candidates) < max_attempts:
        candidates.append(frame)
        frame -= step
    return candidates


def _pose(record: Mapping[str, Any]) -> np.ndarray:
    pose = np.asarray(record["pose_world"], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose_world must be a finite 4x4 matrix")
    return pose


def pose_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[float, float]:
    """Return translation metres and geodesic rotation degrees."""
    pose_a, pose_b = _pose(a), _pose(b)
    translation = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
    relative = pose_a[:3, :3].T @ pose_b[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    rotation = math.degrees(math.acos(cosine))
    return translation, rotation


def assess_tail_recovery(
    original: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    *,
    overlap_frames: int,
    min_connection_frames: int,
    min_suffix_coverage: float,
    max_trailing_missing: int,
    max_translation_error_m: float,
    max_rotation_error_deg: float,
    max_rotation_alignment_dispersion_deg: float = 5.0,
) -> dict[str, Any]:
    """Check that a recovery bridges the old track and covers its lost suffix."""
    if overlap_frames < 1 or min_connection_frames < 1:
        raise ValueError("overlap frame counts must be positive")
    if not 0.0 <= min_suffix_coverage <= 1.0:
        raise ValueError("min_suffix_coverage must be in [0, 1]")
    old = records_by_frame(original)
    new = records_by_frame(recovery)
    gap = tail_gap(original)
    last_valid, last_frame = gap["last_valid_frame"], gap["last_frame"]
    if last_valid < 0 or last_valid >= last_frame:
        return {"accepted": False, "reason": "original_has_no_recoverable_tail", **gap}

    overlap_start = max(0, last_valid - overlap_frames + 1)
    common = [
        frame for frame in range(overlap_start, last_valid + 1)
        if has_pose(old.get(frame)) and has_pose(new.get(frame))
    ]
    if len(common) < min_connection_frames:
        return {
            "accepted": False,
            "reason": f"connection_frames={len(common)}<{min_connection_frames}",
            "connection_frames": len(common),
            **gap,
        }
    # The beginning of the overlap is deliberately farther from the original
    # failure boundary and is therefore more trustworthy than its last poses.
    common = common[:max(min_connection_frames, min(10, len(common)))]
    distances = [pose_distance(old[frame], new[frame]) for frame in common]
    translation_error = float(np.median([value[0] for value in distances]))
    raw_rotation_error = float(np.median([value[1] for value in distances]))

    # A late FoundPose seed may choose a symmetry-equivalent object frame while
    # reverse GoTrack still recovers the correct relative motion.  Accept that
    # case only when one constant right-multiplied rotation explains every
    # trusted overlap frame.  Time-varying drift cannot pass this test.
    corrections = []
    for frame in common:
        old_rotation = _pose(old[frame])[:3, :3]
        new_rotation = _pose(new[frame])[:3, :3]
        corrections.append(new_rotation.T @ old_rotation)
    u, _, vt = np.linalg.svd(np.mean(corrections, axis=0))
    rotation_alignment = u @ vt
    if np.linalg.det(rotation_alignment) < 0:
        u[:, -1] *= -1
        rotation_alignment = u @ vt
    alignment_errors = []
    aligned_errors = []
    for frame, correction in zip(common, corrections):
        cosine = float(np.clip(
            (np.trace(rotation_alignment.T @ correction) - 1.0) * 0.5, -1.0, 1.0,
        ))
        alignment_errors.append(math.degrees(math.acos(cosine)))
        old_rotation = _pose(old[frame])[:3, :3]
        new_rotation = _pose(new[frame])[:3, :3] @ rotation_alignment
        cosine = float(np.clip(
            (np.trace(old_rotation.T @ new_rotation) - 1.0) * 0.5, -1.0, 1.0,
        ))
        aligned_errors.append(math.degrees(math.acos(cosine)))
    alignment_dispersion = float(np.median(alignment_errors))
    apply_alignment = (
        raw_rotation_error > max_rotation_error_deg
        and alignment_dispersion <= max_rotation_alignment_dispersion_deg
    )
    rotation_error = float(np.median(aligned_errors)) if apply_alignment else raw_rotation_error

    suffix_frames = list(range(last_valid + 1, last_frame + 1))
    suffix_valid = sum(has_pose(new.get(frame)) for frame in suffix_frames)
    suffix_coverage = suffix_valid / len(suffix_frames) if suffix_frames else 1.0
    recovery_last_valid = max(
        (frame for frame in suffix_frames if has_pose(new.get(frame))),
        default=last_valid,
    )
    trailing_missing = last_frame - recovery_last_valid

    reasons: list[str] = []
    if translation_error > max_translation_error_m:
        reasons.append(f"translation_error_m={translation_error:.4f}>{max_translation_error_m:.4f}")
    if rotation_error > max_rotation_error_deg:
        reasons.append(f"rotation_error_deg={rotation_error:.2f}>{max_rotation_error_deg:.2f}")
        if raw_rotation_error > max_rotation_error_deg:
            reasons.append(
                f"rotation_alignment_dispersion_deg={alignment_dispersion:.2f}"
                f">{max_rotation_alignment_dispersion_deg:.2f}"
            )
    if suffix_coverage < min_suffix_coverage:
        reasons.append(f"suffix_coverage={suffix_coverage:.3f}<{min_suffix_coverage:.3f}")
    if trailing_missing > max_trailing_missing:
        reasons.append(f"trailing_missing={trailing_missing}>{max_trailing_missing}")
    return {
        "accepted": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
        "connection_frames": len(common),
        "median_translation_error_m": translation_error,
        "median_rotation_error_deg": rotation_error,
        "median_raw_rotation_error_deg": raw_rotation_error,
        "rotation_alignment_applied": apply_alignment,
        "rotation_alignment_dispersion_deg": alignment_dispersion,
        "rotation_alignment": rotation_alignment.tolist() if apply_alignment else None,
        "suffix_frames": len(suffix_frames),
        "suffix_valid_poses": suffix_valid,
        "suffix_coverage": suffix_coverage,
        "trailing_missing_after_recovery": trailing_missing,
        **gap,
    }


def merge_tail_recovery(
    original: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    *,
    recovery_seed_frame: int,
    rotation_alignment: Sequence[Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    """Preserve the trusted prefix and replace only the previously lost tail."""
    old = records_by_frame(original)
    new = records_by_frame(recovery)
    gap = tail_gap(original)
    last_valid = gap["last_valid_frame"]
    if last_valid < 0:
        raise ValueError("original records contain no valid pose")
    alignment = None if rotation_alignment is None else np.asarray(rotation_alignment, dtype=np.float64)
    if alignment is not None and alignment.shape != (3, 3):
        raise ValueError("rotation_alignment must be 3x3")
    merged: list[dict[str, Any]] = []
    for frame in sorted(set(old) | set(new)):
        source = old if frame <= last_valid else new
        row = dict(source.get(frame, old.get(frame, new.get(frame, {"frame_index": frame}))))
        row["frame_index"] = frame
        if frame > last_valid:
            if alignment is not None and has_pose(row):
                pose = _pose(row).copy()
                pose[:3, :3] = pose[:3, :3] @ alignment
                row["pose_world"] = pose.tolist()
                row["rotation_world"] = pose[:3, :3].tolist()
            row["tail_recovery_seed_frame"] = int(recovery_seed_frame)
            row["tail_recovery_source"] = row.get("tracking_direction", "recovery")
        merged.append(row)
    return merged
