#!/usr/bin/env python3
"""Export promoted v2 pose archives as isolated, temporally smoothed v3 files.

The clean capture tree is read-only. Outputs mirror
``<campaign>/<object>/<episode>`` below a separate object-tracking root, so a
campaign does not need a symlink workspace. Only episodes that already contain
``object_6d_pose_v2.npz`` are eligible.

The default is an audit. Pass ``--write`` to atomically create the v3 archives
and a central provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.process.smooth_gotrack_pose_records import _gated_gaussian_segment


DEFAULT_SOURCE_ROOT = Path.home() / "shared_data/capture/eccv2026/v0"
DEFAULT_OUTPUT_ROOT = Path.home() / "shared_data/object_tracking/object_6d_pose_v3"
DEFAULT_CAMPAIGNS = ("human", "inspire_dftp", "inspire_f1", "allegro_v5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pose_archive(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        indexed: list[tuple[int, str]] = []
        for key in archive.files:
            if not key.startswith("frame_"):
                raise ValueError(f"unexpected key {key!r}")
            try:
                indexed.append((int(key.removeprefix("frame_")), key))
            except ValueError as exc:
                raise ValueError(f"invalid frame key {key!r}") from exc
        indexed.sort()
        indices = [index for index, _ in indexed]
        if indices != list(range(len(indices))):
            raise ValueError("frame keys are not contiguous from frame_0")
        keys = [key for _, key in indexed]
        poses = np.asarray([archive[key] for key in keys], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"poses are not finite Nx4x4 matrices: {poses.shape}")
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(poses[:, 3, :], expected_last_row, atol=1e-5):
        raise ValueError("invalid homogeneous last row")
    return keys, poses


def _atomic_savez(path: Path, keys: list[str], poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        payload = {key: pose.astype(np.float32) for key, pose in zip(keys, poses)}
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _correction_summary(original: np.ndarray, filtered: np.ndarray) -> dict[str, float]:
    translation = np.linalg.norm(
        filtered[:, :3, 3] - original[:, :3, 3], axis=1
    ) * 1000.0
    original_rotation = Rotation.from_matrix(original[:, :3, :3])
    filtered_rotation = Rotation.from_matrix(filtered[:, :3, :3])
    rotation = np.degrees((original_rotation.inv() * filtered_rotation).magnitude())
    return {
        "translation_mm_p50": float(np.percentile(translation, 50)),
        "translation_mm_p95": float(np.percentile(translation, 95)),
        "translation_mm_max": float(translation.max()),
        "rotation_deg_p50": float(np.percentile(rotation, 50)),
        "rotation_deg_p95": float(np.percentile(rotation, 95)),
        "rotation_deg_max": float(rotation.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--campaign",
        action="append",
        choices=DEFAULT_CAMPAIGNS,
        help="Repeat to select campaigns. Default: all four promoted campaigns.",
    )
    parser.add_argument("--input-name", default="object_6d_pose_v2.npz")
    parser.add_argument("--output-name", default="object_6d_pose_v3.npz")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--gate-translation-mm-s", type=float, default=15.0)
    parser.add_argument("--gate-rotation-deg-s", type=float, default=7.5)
    parser.add_argument("--gate-motion-span-frames", type=int, default=11)
    parser.add_argument("--gate-min-run-frames", type=int, default=15)
    parser.add_argument("--gate-transition-frames", type=int, default=3)
    parser.add_argument("--gate-hard-translation-mm-s", type=float, default=300.0)
    parser.add_argument("--gate-hard-rotation-deg-s", type=float, default=300.0)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    campaigns = tuple(args.campaign or DEFAULT_CAMPAIGNS)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if Path(args.input_name).name != args.input_name or Path(args.output_name).name != args.output_name:
        raise ValueError("--input-name and --output-name must be plain file names")
    if args.window_size < 1 or args.window_size % 2 != 1 or args.sigma <= 0.0:
        raise ValueError("--window-size must be positive and odd; --sigma must be positive")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.gate_motion_span_frames < 3 or args.gate_motion_span_frames % 2 != 1:
        raise ValueError("--gate-motion-span-frames must be odd and at least 3")
    if args.gate_min_run_frames < 1 or args.gate_transition_frames < 0:
        raise ValueError("gate run length must be positive and transition non-negative")

    sources: list[tuple[str, Path]] = []
    campaign_inventory: dict[str, dict[str, int]] = {}
    for campaign in campaigns:
        campaign_root = source_root / campaign
        if not campaign_root.is_dir():
            raise FileNotFoundError(campaign_root)
        episodes = sum(1 for path in campaign_root.glob("*/*") if path.is_dir())
        files = sorted(campaign_root.glob(f"*/*/{args.input_name}"))
        campaign_inventory[campaign] = {
            "episodes": episodes,
            "eligible_v2": len(files),
            "without_v2": episodes - len(files),
        }
        sources.extend((campaign, path) for path in files)

    parameters = {
        "method": "gated-gaussian",
        "fps": args.fps,
        "window_size": args.window_size,
        "sigma_frames": args.sigma,
        "translation_threshold_mm_s": args.gate_translation_mm_s,
        "rotation_threshold_deg_s": args.gate_rotation_deg_s,
        "motion_span_frames": args.gate_motion_span_frames,
        "minimum_enabled_run_frames": args.gate_min_run_frames,
        "transition_frames": args.gate_transition_frames,
        "hard_translation_threshold_mm_s": args.gate_hard_translation_mm_s,
        "hard_rotation_threshold_deg_s": args.gate_hard_rotation_deg_s,
    }
    print(json.dumps({
        "mode": "write" if args.write else "audit",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "campaigns": campaign_inventory,
        "eligible_total": len(sources),
        "parameters": parameters,
    }, indent=2))
    if not args.write:
        return 0

    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {manifest_path}; pass --overwrite")
    entries: list[dict[str, Any]] = []
    counts = {"written": 0, "error": 0}
    radius = args.window_size // 2
    for position, (campaign, source) in enumerate(sources, start=1):
        object_name = source.parent.parent.name
        episode = source.parent.name
        output = output_root / campaign / object_name / episode / args.output_name
        entry: dict[str, Any] = {
            "campaign": campaign,
            "object_name": object_name,
            "episode": episode,
            "source": str(source),
            "output": str(output),
        }
        try:
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"output exists: {output}")
            keys, original = _load_pose_archive(source)
            filtered, translation_enabled, rotation_enabled = _gated_gaussian_segment(
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
            _atomic_savez(output, keys, filtered)
            entry.update({
                "status": "written",
                "frames": len(keys),
                "source_sha256": _sha256(source),
                "output_sha256": _sha256(output),
                "translation_gate_fraction": float(translation_enabled.mean()),
                "rotation_gate_fraction": float(rotation_enabled.mean()),
                "both_gate_fraction": float((translation_enabled & rotation_enabled).mean()),
                "correction": _correction_summary(original, filtered),
            })
            counts["written"] += 1
        except Exception as exc:  # keep the batch auditable and finish other episodes
            entry.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            counts["error"] += 1
        entries.append(entry)
        if position == 1 or position % 25 == 0 or position == len(sources):
            print(
                f"[v3] {position}/{len(sources)} written={counts['written']} "
                f"error={counts['error']}"
            )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "input_name": args.input_name,
        "output_name": args.output_name,
        "campaigns": campaign_inventory,
        "parameters": parameters,
        "counts": counts,
        "entries": entries,
    }
    _atomic_write_json(manifest_path, manifest)
    print(f"[v3] manifest={manifest_path}")
    if counts["error"]:
        print(f"[v3] completed with {counts['error']} errors")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
