#!/usr/bin/env python3
"""Create a write-isolated workspace for ECCV Robotiq 2F-85 tracking.

Only episodes whose ``grasp_result.json`` records ``grasp_success=true`` are
included.  Capture inputs remain under ``capture/eccv2026/robotiq_2f85``;
workspace episodes contain relative symlinks, while all generated tracking
outputs are written below ``object_tracking/campaigns/robotiq_2f85/workspace``.

The default is a read-only audit. Pass ``--write`` to create the workspace.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SHARED_ROOT = Path.home() / "shared_data"
DEFAULT_SOURCE_REL = Path("capture/eccv2026/robotiq_2f85")
DEFAULT_WORKSPACE_REL = Path("object_tracking/campaigns/robotiq_2f85/workspace")

REQUIRED_LINKED_NAMES = ("cam_param", "C2R.npy", "raw", "videos", "grasp_result.json")
OPTIONAL_LINKED_NAMES = ("paired_human_episode.json",)
GENERATED_NAMES = (
    "undistorted_video",
    "object_tracking_foundpose_gotrack",
    "foundpose_init",
    "gotrack_tracking",
    "object_6d_pose_v2.npz",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_target(link: Path, target: Path) -> str:
    return os.path.relpath(target, start=link.parent)


def _ensure_relative_symlink(link: Path, target: Path, write: bool) -> None:
    expected = _relative_target(link, target)
    if link.is_symlink():
        if os.readlink(link) != expected:
            raise ValueError(
                f"Symlink target mismatch: {link} -> {os.readlink(link)!r}, "
                f"expected {expected!r}"
            )
        if not link.exists():
            raise FileNotFoundError(f"Broken workspace symlink: {link}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    if write:
        link.symlink_to(expected, target_is_directory=target.is_dir())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--source-rel", type=Path, default=DEFAULT_SOURCE_REL)
    parser.add_argument("--workspace-rel", type=Path, default=DEFAULT_WORKSPACE_REL)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument(
        "--allowed-camera-counts",
        nargs="+",
        type=int,
        default=(22, 23),
        help="Allowed matching intrinsic/extrinsic and AVI counts.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    for value in (args.source_rel, args.workspace_rel):
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("Relative roots must not be absolute or contain '..'")
    if any(count <= 0 for count in args.allowed_camera_counts):
        raise ValueError("Allowed camera counts must be positive")

    shared = args.shared_root.expanduser().resolve()
    source = shared / args.source_rel
    workspace = shared / args.workspace_rel
    if not source.is_dir():
        raise FileNotFoundError(source)
    if workspace == source or source in workspace.parents:
        raise ValueError("Workspace must be outside the source dataset")

    object_filter = set(args.objects or [])
    episode_filter = set(args.episodes or [])
    planned = 0
    skipped_unsuccessful = 0
    camera_histogram: dict[str, int] = {}
    object_names: set[str] = set()

    for object_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        if object_filter and object_dir.name not in object_filter:
            continue
        for episode_dir in sorted(
            (path for path in object_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        ):
            if episode_filter and episode_dir.name not in episode_filter:
                continue
            grasp_path = episode_dir / "grasp_result.json"
            if not grasp_path.is_file() or not bool(_read_json(grasp_path).get("grasp_success")):
                skipped_unsuccessful += 1
                continue

            tag = f"{object_dir.name}/{episode_dir.name}"
            generated = [name for name in GENERATED_NAMES if (episode_dir / name).exists()]
            if generated:
                raise ValueError(f"Generated outputs in source episode {tag}: {generated}")
            for name in REQUIRED_LINKED_NAMES:
                if not (episode_dir / name).exists():
                    raise FileNotFoundError(episode_dir / name)

            intrinsics = _read_json(episode_dir / "cam_param/intrinsics.json")
            extrinsics = _read_json(episode_dir / "cam_param/extrinsics.json")
            camera_ids = sorted(set(intrinsics) & set(extrinsics))
            if set(intrinsics) != set(extrinsics) or not camera_ids:
                raise ValueError(f"Intrinsic/extrinsic camera mismatch: {tag}")
            if len(camera_ids) not in args.allowed_camera_counts:
                raise ValueError(
                    f"{tag}: camera count {len(camera_ids)} is not in "
                    f"{sorted(args.allowed_camera_counts)}"
                )
            video_ids = sorted(path.stem for path in (episode_dir / "videos").glob("*.avi"))
            if video_ids != camera_ids:
                raise ValueError(f"{tag}: calibrated camera and AVI sets differ")
            empty_videos = [
                camera_id for camera_id in camera_ids
                if (episode_dir / "videos" / f"{camera_id}.avi").stat().st_size == 0
            ]
            if empty_videos:
                raise ValueError(f"{tag}: empty videos {empty_videos}")

            destination = workspace / object_dir.name / episode_dir.name
            if args.write:
                destination.mkdir(parents=True, exist_ok=True)
            for name in REQUIRED_LINKED_NAMES:
                _ensure_relative_symlink(destination / name, episode_dir / name, args.write)
            linked_optional_names = []
            for name in OPTIONAL_LINKED_NAMES:
                target = episode_dir / name
                if target.exists():
                    _ensure_relative_symlink(destination / name, target, args.write)
                    linked_optional_names.append(name)

            manifest = {
                "schema_version": 1,
                "source_episode_rel": str(episode_dir.relative_to(shared)),
                "workspace_episode_rel": str(destination.relative_to(shared)),
                "object_name": object_dir.name,
                "episode": episode_dir.name,
                "grasp_success": True,
                "fixed_camera_ids": camera_ids,
                "camera_count": len(camera_ids),
                "source_video_directory": "videos",
                "pipeline_video_directory": "videos",
                "required_linked_names": list(REQUIRED_LINKED_NAMES),
                "optional_linked_names": linked_optional_names,
            }
            manifest_path = destination / "source_episode.json"
            if manifest_path.exists():
                if _read_json(manifest_path) != manifest:
                    raise ValueError(f"Workspace manifest mismatch: {manifest_path}")
            elif args.write:
                _atomic_json(manifest_path, manifest)

            planned += 1
            object_names.add(object_dir.name)
            key = str(len(camera_ids))
            camera_histogram[key] = camera_histogram.get(key, 0) + 1

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "write" if args.write else "dry-run",
        "source_root": str(source),
        "workspace_root": str(workspace),
        "objects": len(object_names),
        "eligible_success_episodes": planned,
        "skipped_unsuccessful_or_unlabelled": skipped_unsuccessful,
        "camera_count_histogram": camera_histogram,
        "source_dataset_modified": False,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.report is not None:
        _atomic_json(args.report.expanduser().resolve(), summary)
        print(f"report={args.report.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
