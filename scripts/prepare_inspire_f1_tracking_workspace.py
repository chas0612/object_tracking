#!/usr/bin/env python3
"""Create a write-isolated workspace for ECCV Inspire F1 tracking.

The clean source tree remains under ``capture/eccv2026/v0/inspire_f1``.
Every workspace episode contains relative symlinks to immutable capture inputs:

* 21-camera calibration and only those 21 calibrated videos;
* ``C2R.npy`` and robot ``raw/`` streams;
* the original ``object_6d/`` text trajectory;
* ``grasp_result.json`` when the capture provides it.

The two videos absent from the fixed-camera calibration are intentionally not
linked. FoundPose, SAM3, GoTrack, and debug outputs can then be written beside
the links without modifying the clean dataset. The original ``object_6d`` text
sequence remains the final export frame contract.

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
DEFAULT_SOURCE_REL = Path("capture/eccv2026/v0/inspire_f1")
DEFAULT_WORKSPACE_REL = Path("object_tracking/campaigns/inspire_f1/workspace")

REQUIRED_LINKED_NAMES = (
    "cam_param",
    "C2R.npy",
    "raw",
    "object_6d",
)

OPTIONAL_LINKED_NAMES = (
    "grasp_result.json",
)

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
    parser.add_argument("--expected-cameras", type=int, default=21)
    parser.add_argument(
        "--max-missing-calibrated-videos",
        type=int,
        default=1,
        help="Allow this many calibrated cameras without a published video per episode.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    for value in (args.source_rel, args.workspace_rel):
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("Relative roots must not be absolute or contain '..'")
    if args.expected_cameras < 0 or args.max_missing_calibrated_videos < 0:
        raise ValueError("Camera counts must be non-negative")

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
    video_links = 0
    excluded_videos = 0
    missing_calibrated_videos = 0
    object_names: set[str] = set()

    for object_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        if object_filter and object_dir.name not in object_filter:
            continue
        for episode_dir in sorted(
            (
                path
                for path in object_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        ):
            if episode_filter and episode_dir.name not in episode_filter:
                continue
            tag = f"{object_dir.name}/{episode_dir.name}"
            generated = [name for name in GENERATED_NAMES if (episode_dir / name).exists()]
            if generated:
                raise ValueError(f"Generated outputs in source episode {tag}: {generated}")

            intrinsics = _read_json(episode_dir / "cam_param/intrinsics.json")
            extrinsics = _read_json(episode_dir / "cam_param/extrinsics.json")
            camera_ids = sorted(set(intrinsics) & set(extrinsics))
            if set(intrinsics) != set(extrinsics) or not camera_ids:
                raise ValueError(f"Intrinsic/extrinsic camera mismatch: {tag}")
            if args.expected_cameras and len(camera_ids) != args.expected_cameras:
                raise ValueError(
                    f"{tag}: expected {args.expected_cameras} calibrated cameras, "
                    f"found {len(camera_ids)}"
                )

            source_videos = {
                path.stem: path for path in (episode_dir / "vid").glob("*.mp4")
            }
            missing_videos = sorted(set(camera_ids) - set(source_videos))
            if len(missing_videos) > args.max_missing_calibrated_videos:
                raise FileNotFoundError(
                    f"{tag}: missing calibrated videos {missing_videos}; "
                    f"limit={args.max_missing_calibrated_videos}"
                )
            linked_camera_ids = sorted(set(camera_ids) - set(missing_videos))
            excluded_ids = sorted(set(source_videos) - set(camera_ids))

            destination = workspace / object_dir.name / episode_dir.name
            videos = destination / "videos"
            if args.write:
                videos.mkdir(parents=True, exist_ok=True)

            for name in REQUIRED_LINKED_NAMES:
                target = episode_dir / name
                if not target.exists():
                    raise FileNotFoundError(target)
                _ensure_relative_symlink(destination / name, target, args.write)

            linked_optional_names = []
            for name in OPTIONAL_LINKED_NAMES:
                target = episode_dir / name
                if target.exists():
                    _ensure_relative_symlink(destination / name, target, args.write)
                    linked_optional_names.append(name)

            for camera_id in linked_camera_ids:
                source_video = source_videos[camera_id]
                if source_video.stat().st_size == 0:
                    raise ValueError(f"Empty source video: {source_video}")
                _ensure_relative_symlink(
                    videos / source_video.name, source_video, args.write
                )
                video_links += 1

            manifest = {
                "schema_version": 1,
                "source_episode_rel": str(episode_dir.relative_to(shared)),
                "workspace_episode_rel": str(destination.relative_to(shared)),
                "object_name": object_dir.name,
                "episode": episode_dir.name,
                "calibrated_camera_ids": camera_ids,
                "fixed_camera_ids": linked_camera_ids,
                "missing_calibrated_video_ids": missing_videos,
                "source_video_directory": "vid",
                "pipeline_video_directory": "videos",
                "excluded_video_ids": excluded_ids,
                "required_linked_names": list(REQUIRED_LINKED_NAMES),
                "optional_linked_names": linked_optional_names,
                "final_pose_frame_contract": "object_6d",
            }
            manifest_path = destination / "source_episode.json"
            if manifest_path.exists():
                if _read_json(manifest_path) != manifest:
                    raise ValueError(f"Workspace manifest mismatch: {manifest_path}")
            elif args.write:
                _atomic_json(manifest_path, manifest)

            planned += 1
            excluded_videos += len(excluded_ids)
            missing_calibrated_videos += len(missing_videos)
            object_names.add(object_dir.name)

    mode = "write" if args.write else "dry-run"
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "source_root": str(source),
        "workspace_root": str(workspace),
        "objects": len(object_names),
        "episodes": planned,
        "fixed_video_links": video_links,
        "excluded_videos": excluded_videos,
        "missing_calibrated_videos": missing_calibrated_videos,
        "expected_cameras": args.expected_cameras,
        "max_missing_calibrated_videos": args.max_missing_calibrated_videos,
        "required_linked_names": list(REQUIRED_LINKED_NAMES),
        "optional_linked_names": list(OPTIONAL_LINKED_NAMES),
        "source_dataset_modified": False,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        _atomic_json(report_path, summary)
        print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
