#!/usr/bin/env python3
"""Compact archived FoundPose+GoTrack attempts without losing trajectories.

The archive contains large, reproducible per-camera intermediates.  This tool
keeps the merged and directional ``world_pose_records.json`` trajectories,
FoundPose initialization evidence, manifests, summaries, and small init-pose
files.  It removes source-frame copies, masks, and dense per-camera tracking
records.

The default is a read-only audit. Pass ``--apply`` for in-place deletion.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = (
    Path.home() / "shared_data/object_tracking/foundpose_gotrack_archive"
)
DEFAULT_REPORT_ROOT = Path.home() / "shared_data/object_tracking/exports"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _keep(relative: Path) -> tuple[bool, str]:
    parts = relative.parts
    name = relative.name

    if name == "world_pose_records.json":
        return True, "trajectory"
    if "foundpose_init" in parts:
        return True, "foundpose_initialization"
    if name in {
        "metadata.json",
        "run_manifest.json",
        "merge_manifest.json",
        "summary.json",
        "anchor_bank_summary.json",
        "anchor_bank.summary.json",
        "anchor_debug_frames.json",
        "candidate_bank.json",
        "result.json",
    }:
        return True, "metadata"
    if name == "anchor_bank.npz":
        return True, "compact_anchor_evidence"
    if "init_poses" in parts and name.endswith((".json", ".npy", ".npz")):
        return True, "initial_pose"
    if name.startswith(".archive_compaction_") and name.endswith(".json.gz"):
        return True, "prior_compaction_report"
    return False, "reproducible_dense_intermediate"


def _scan(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        keep, reason = _keep(relative)
        record = {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "reason": reason,
        }
        (kept if keep else removed).append(record)
    return kept, removed


def _remove_empty_directories(root: Path) -> int:
    removed = 0
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.archive_root.expanduser().resolve()
    expected = DEFAULT_ROOT.resolve()
    if root != expected:
        raise ValueError(
            f"Refusing a non-canonical archive root: {root} (expected {expected})"
        )
    if not root.is_dir():
        raise FileNotFoundError(root)

    kept, removed = _scan(root)
    before_bytes = sum(item["bytes"] for item in kept + removed)
    kept_bytes = sum(item["bytes"] for item in kept)
    removed_bytes = sum(item["bytes"] for item in removed)
    print(
        f"mode={'apply' if args.apply else 'dry-run'} files={len(kept) + len(removed)} "
        f"before={before_bytes} keep={kept_bytes} remove={removed_bytes}",
        flush=True,
    )
    print(
        f"estimated_reduction={removed_bytes / before_bytes:.1%}"
        if before_bytes
        else "estimated_reduction=0.0%",
        flush=True,
    )

    if not args.apply:
        return 0

    report_root = args.report_root.expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_root / f"foundpose_gotrack_archive_compaction_{stamp}.json.gz"
    if report_path.exists():
        raise FileExistsError(report_path)

    report = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "archive_root": str(root),
        "policy": {
            "kept": [
                "world_pose_records.json",
                "foundpose_init/**",
                "run/merge manifests and compact summaries",
                "anchor_bank.npz",
                "init_poses/**",
            ],
            "removed": [
                "copied source frames and masks",
                "per-camera frame_poses",
                "singleview_results",
                "other reproducible dense intermediates",
            ],
        },
        "before_bytes": before_bytes,
        "kept_bytes": kept_bytes,
        "removed_bytes": removed_bytes,
        "kept_files": kept,
        "removed_files": removed,
    }
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    with gzip.open(temporary_report, "wt", encoding="utf-8") as stream:
        json.dump(report, stream, separators=(",", ":"))
    os.replace(temporary_report, report_path)

    deleted_bytes = 0
    for item in removed:
        path = root / item["path"]
        path.unlink()
        deleted_bytes += int(item["bytes"])
    empty_directories = _remove_empty_directories(root)
    print(
        f"deleted_files={len(removed)} deleted_bytes={deleted_bytes} "
        f"empty_directories={empty_directories}",
        flush=True,
    )
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
