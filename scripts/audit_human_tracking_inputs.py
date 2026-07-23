#!/usr/bin/env python3
"""Audit ECCV human source data, isolated workspace, meshes, and caches.

This command is read-only unless ``--report`` is provided.  It verifies that
the pipeline workspace mirrors every source episode without placing generated
outputs in the clean ``v0/human`` dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SHARED = Path.home() / "shared_data"
DEFAULT_SOURCE_REL = Path("capture/eccv2026/v0/human")
DEFAULT_WORKSPACE_REL = Path("object_tracking/campaigns/human/workspace")
DEFAULT_MESH_REL = Path("mesh_new")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _episodes(root: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for object_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for episode_dir in sorted(
            (
                path for path in object_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        ):
            result[(object_dir.name, episode_dir.name)] = episode_dir
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--source-rel", type=Path, default=DEFAULT_SOURCE_REL)
    parser.add_argument("--workspace-rel", type=Path, default=DEFAULT_WORKSPACE_REL)
    parser.add_argument("--mesh-root-rel", type=Path, default=DEFAULT_MESH_REL)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--deep-frame-check",
        action="store_true",
        help="Also enumerate every object-pose and MANO frame (slow on NAS).",
    )
    parser.add_argument(
        "--allow-missing-cache",
        action="store_true",
        help="Return success when structure is valid but onboarding is incomplete.",
    )
    args = parser.parse_args()

    for value in (args.source_rel, args.workspace_rel, args.mesh_root_rel):
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("Dataset roots must be safe relative paths")

    shared = args.shared_root.expanduser().resolve()
    source = shared / args.source_rel
    workspace = shared / args.workspace_rel
    mesh_root = shared / args.mesh_root_rel
    if not source.is_dir() or not workspace.is_dir() or not mesh_root.is_dir():
        raise FileNotFoundError("source, workspace, and mesh root must exist")

    source_episodes = _episodes(source)
    workspace_episodes = _episodes(workspace)
    errors: list[str] = []
    warnings: list[str] = []
    if set(source_episodes) != set(workspace_episodes):
        errors.append(
            f"episode set mismatch: source_only={sorted(set(source_episodes) - set(workspace_episodes))} "
            f"workspace_only={sorted(set(workspace_episodes) - set(source_episodes))}"
        )

    object_names = sorted({key[0] for key in source_episodes})
    missing_mesh: list[str] = []
    missing_cache: list[str] = []
    for object_name in object_names:
        mesh = mesh_root / object_name / f"{object_name}.obj"
        repre = (
            mesh_root
            / object_name
            / "foundpose_assets/object_repre/v1"
            / object_name
            / "1/repre.pth"
        )
        if not mesh.is_file():
            missing_mesh.append(object_name)
        if not repre.is_file() or repre.stat().st_size < 1_000_000:
            missing_cache.append(object_name)

    checked_video_links = 0
    frame_count_mismatches: list[str] = []
    for key, source_episode in source_episodes.items():
        tag = f"{key[0]}/{key[1]}"
        workspace_episode = workspace_episodes.get(key)
        if workspace_episode is None:
            continue
        try:
            intrinsics = _read_json(source_episode / "cam_param/intrinsics.json")
            extrinsics = _read_json(source_episode / "cam_param/extrinsics.json")
            fixed_ids = sorted(set(intrinsics) & set(extrinsics))
            if set(intrinsics) != set(extrinsics) or len(fixed_ids) != 21:
                errors.append(f"{tag}: expected matching 21-camera calibration")
                continue
            c2r = np.load(source_episode / "C2R.npy", allow_pickle=False)
            if c2r.shape != (4, 4) or not np.isfinite(c2r).all():
                errors.append(f"{tag}: invalid C2R")
            manifest = _read_json(workspace_episode / "source_episode.json")
            if manifest.get("fixed_camera_ids") != fixed_ids:
                errors.append(f"{tag}: workspace manifest camera mismatch")
            for name in ("cam_param", "C2R.npy", "hand", "object_6d"):
                link = workspace_episode / name
                if not link.is_symlink() or not link.exists():
                    errors.append(f"{tag}: invalid workspace link {name}")
            videos = sorted((workspace_episode / "videos").glob("*.mp4"))
            if [path.stem for path in videos] != fixed_ids:
                errors.append(f"{tag}: workspace video set mismatch")
            for video in videos:
                checked_video_links += 1
                if (
                    not video.is_symlink()
                    or os.path.isabs(os.readlink(video))
                    or not video.is_file()
                    or video.stat().st_size == 0
                ):
                    errors.append(f"{tag}: invalid video link {video.name}")
            generated = [
                name for name in (
                    "undistorted_video",
                    "object_tracking_foundpose_gotrack",
                    "foundpose_init",
                )
                if (source_episode / name).exists()
            ]
            if generated:
                errors.append(f"{tag}: generated outputs in source: {generated}")
            if args.deep_frame_check:
                pose_count = len(list((source_episode / "object_6d").glob("pose_*.txt")))
                mano_count = len(list((source_episode / "hand/mano").glob("*.obj")))
                param_count = len(list((source_episode / "hand/mano_params").glob("*")))
                if not (pose_count == mano_count == param_count and pose_count > 0):
                    frame_count_mismatches.append(
                        f"{tag}: object={pose_count} mano={mano_count} params={param_count}"
                    )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{tag}: {type(exc).__name__}: {exc}")

    if missing_mesh:
        errors.append(f"missing meshes: {missing_mesh}")
    if frame_count_mismatches:
        errors.extend(frame_count_mismatches)
    if missing_cache:
        warnings.append(f"missing onboarding cache: {missing_cache}")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "workspace_root": str(workspace),
        "mesh_root": str(mesh_root),
        "objects": len(object_names),
        "episodes": len(source_episodes),
        "checked_fixed_video_links": checked_video_links,
        "deep_frame_check": args.deep_frame_check,
        "missing_mesh": missing_mesh,
        "missing_cache": missing_cache,
        "errors": errors,
        "warnings": warnings,
        "ready_for_tracking": not errors and not missing_cache,
    }
    print(
        json.dumps(
            {
                "objects": report["objects"],
                "episodes": report["episodes"],
                "checked_fixed_video_links": checked_video_links,
                "missing_mesh": len(missing_mesh),
                "missing_cache": len(missing_cache),
                "errors": len(errors),
                "ready_for_tracking": report["ready_for_tracking"],
            },
            indent=2,
        )
    )
    for warning in warnings:
        print(f"[warning] {warning}")
    for error in errors:
        print(f"[error] {error}")
    if args.report is not None:
        _atomic_json(args.report.expanduser().resolve(), report)
        print(f"report={args.report.expanduser().resolve()}")
    if errors:
        return 1
    if missing_cache and not args.allow_missing_cache:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
