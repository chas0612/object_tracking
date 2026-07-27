#!/usr/bin/env python3
"""Create a write-isolated tracking workspace for the ECCV allegro_v5 captures.

Source captures remain untouched.  Each workspace episode contains relative
symlinks to calibration, C2R, and the raw robot streams, plus the 22 videos
named by the fixed-camera intrinsic/extrinsic calibration.  Generated FoundPose
and GoTrack outputs can therefore live beside these links without writing into
``capture/eccv2026/v0/allegro_v5``.

This is the allegro counterpart of ``prepare_human_tracking_workspace.py``.  The
robot captures differ from the human ones in three ways:

* 22 calibrated cameras and no ego videos, against 21 plus two ego streams.
* Per-episode payload is ``raw/`` (arm, hand, timestamps); there is no ``hand/``
  MANO directory and no ``object_6d/`` text pose sequence.
* The existing ``object_6d_pose.npz`` is **not** the frame contract for this
  campaign.  It was tracked before the published ``vid/*.mp4`` re-encode dropped
  the first three frames, so it sits three frames out of step with the videos.
  The new poses are defined against the videos instead; export with
  ``export_latest_gotrack_poses.py --frame-contract video``.

This script therefore does no frame accounting — it builds the links and
validates the camera set.  The export step counts the videos it aligns to, which
is the only place that count has to be right.

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
DEFAULT_SOURCE_REL = Path("capture/eccv2026/v0/allegro_v5")
DEFAULT_WORKSPACE_REL = Path("object_tracking/campaigns/allegro_v5/workspace")

# Linked into every workspace episode. ``raw`` carries the arm/hand/timestamp
# streams; keep it reachable from the workspace even though the tracking stages
# themselves only read cam_param and videos.
LINKED_NAMES = ("cam_param", "C2R.npy", "raw")

# Names that only ever appear as pipeline output. Their presence in the source
# tree means something already wrote into the clean dataset.
GENERATED_NAMES = (
    "undistorted_video",
    "object_tracking_foundpose_gotrack",
    "foundpose_init",
    "gotrack_tracking",
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
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    if write:
        link.symlink_to(expected, target_is_directory=target.is_dir())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--source-rel", type=Path, default=DEFAULT_SOURCE_REL)
    parser.add_argument("--workspace-rel", type=Path, default=DEFAULT_WORKSPACE_REL)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--expected-cameras", type=int, default=22,
                        help="Required calibrated camera count per episode. 0 disables the check.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional JSON summary written even in dry-run mode.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    for value in (args.source_rel, args.workspace_rel):
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("Relative roots must not be absolute or contain '..'")

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
    for object_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        if object_filter and object_dir.name not in object_filter:
            continue
        for episode_dir in sorted(
            (path for path in object_dir.iterdir() if path.is_dir() and path.name.isdigit()),
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

            destination = workspace / object_dir.name / episode_dir.name
            videos = destination / "videos"
            if args.write:
                videos.mkdir(parents=True, exist_ok=True)

            for name in LINKED_NAMES:
                target = episode_dir / name
                if not target.exists():
                    raise FileNotFoundError(target)
                _ensure_relative_symlink(destination / name, target, args.write)

            for camera_id in camera_ids:
                source_video = episode_dir / "vid" / f"{camera_id}.mp4"
                if not source_video.is_file() or source_video.stat().st_size == 0:
                    raise FileNotFoundError(source_video)
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
                "fixed_camera_ids": camera_ids,
                "source_video_directory": "vid",
                "pipeline_video_directory": "videos",
                "linked_names": list(LINKED_NAMES),
            }
            manifest_path = destination / "source_episode.json"
            if manifest_path.exists():
                if _read_json(manifest_path) != manifest:
                    raise ValueError(f"Workspace manifest mismatch: {manifest_path}")
            elif args.write:
                _atomic_json(manifest_path, manifest)
            planned += 1

    mode = "write" if args.write else "dry-run"
    print(
        f"mode={mode} source={source} workspace={workspace} "
        f"episodes={planned} fixed_video_links={video_links}",
        flush=True,
    )
    if args.write:
        print("source_dataset_modified=false", flush=True)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(report_path, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "source_root": str(source),
            "workspace_root": str(workspace),
            "episodes": planned,
            "fixed_video_links": video_links,
            "linked_names": list(LINKED_NAMES),
            "expected_cameras": args.expected_cameras,
        })
        print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
