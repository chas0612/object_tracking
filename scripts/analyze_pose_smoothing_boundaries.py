#!/usr/bin/env python3
"""Measure pose-step artifacts where an offline smoothing sidecar turns on/off."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _load(path: Path) -> np.ndarray:
    rows = json.loads(path.read_text(encoding="utf-8"))
    indexed = sorted(
        (int(row["frame_index"]), np.asarray(row["pose_world"], dtype=np.float64))
        for row in rows
        if isinstance(row, dict) and row.get("pose_world") is not None
    )
    if not indexed or [item[0] for item in indexed] != list(range(len(indexed))):
        raise ValueError(f"Expected contiguous poses from frame zero: {path}")
    return np.asarray([item[1] for item in indexed])


def _component_report(raw: np.ndarray, filtered: np.ndarray, component: str) -> dict:
    if component == "translation":
        correction = np.linalg.norm(filtered[:, :3, 3] - raw[:, :3, 3], axis=1) * 1000.0
        raw_step = np.diff(raw[:, :3, 3], axis=0)
        filtered_step = np.diff(filtered[:, :3, 3], axis=0)
        step_artifact = np.linalg.norm(filtered_step - raw_step, axis=1) * 1000.0
        unit = "mm"
        epsilon = 1e-7
    else:
        raw_rotation = Rotation.from_matrix(raw[:, :3, :3])
        filtered_rotation = Rotation.from_matrix(filtered[:, :3, :3])
        correction = np.degrees((raw_rotation.inv() * filtered_rotation).magnitude())
        raw_step = raw_rotation[:-1].inv() * raw_rotation[1:]
        filtered_step = filtered_rotation[:-1].inv() * filtered_rotation[1:]
        step_artifact = np.degrees((raw_step.inv() * filtered_step).magnitude())
        unit = "deg"
        epsilon = 1e-9
    applied = correction > epsilon
    boundary_frames = np.flatnonzero(applied[1:] != applied[:-1]) + 1
    artifacts = step_artifact[boundary_frames - 1]
    order = np.argsort(artifacts)[::-1]
    return {
        "unit": unit,
        "applied_fraction": float(applied.mean()),
        "boundaries": int(len(boundary_frames)),
        "boundary_artifact_p50": float(np.percentile(artifacts, 50)) if len(artifacts) else 0.0,
        "boundary_artifact_p95": float(np.percentile(artifacts, 95)) if len(artifacts) else 0.0,
        "boundary_artifact_max": float(artifacts.max()) if len(artifacts) else 0.0,
        "largest_boundaries": [
            {
                "frame": int(boundary_frames[index]),
                "turns_on": bool(applied[boundary_frames[index]]),
                "step_artifact": float(artifacts[index]),
                "correction_before": float(correction[boundary_frames[index] - 1]),
                "correction_after": float(correction[boundary_frames[index]]),
            }
            for index in order[:8]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--filtered", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    raw_path = Path(args.raw).expanduser().resolve()
    filtered_path = Path(args.filtered).expanduser().resolve()
    raw = _load(raw_path)
    filtered = _load(filtered_path)
    if raw.shape != filtered.shape:
        raise ValueError(f"Pose shapes differ: {raw.shape} != {filtered.shape}")
    result = {
        "label": args.label,
        "raw": str(raw_path),
        "filtered": str(filtered_path),
        "frames": len(raw),
        "translation": _component_report(raw, filtered, "translation"),
        "rotation": _component_report(raw, filtered, "rotation"),
    }
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
