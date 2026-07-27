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
* The frame contract is ``object_6d_pose.npz``, whose length follows
  ``raw/timestamps/frame_id.npy``.  The published ``vid/*.mp4`` re-encode dropped
  the first three frames of the original capture, so the contract is longer than
  the tracked videos and record ``k`` is contract frame ``k + 3``.  This script
  verifies that offset per episode and records it in the manifest; feed the same
  value to ``export_latest_gotrack_poses.py --frame-index-offset``.

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
# streams that define the frame contract, so keep it reachable from the
# workspace even though the tracking stages themselves only read cam_param and
# videos.
LINKED_NAMES = ("cam_param", "C2R.npy", "raw")

# Names that only ever appear as pipeline output. Their presence in the source
# tree means something already wrote into the clean dataset.
GENERATED_NAMES = (
    "undistorted_video",
    "object_tracking_foundpose_gotrack",
    "foundpose_init",
    "gotrack_tracking",
)

# Manifest keys that describe workspace structure. A re-run must reproduce these
# exactly; the frame-contract keys are refreshed instead, because they are only
# populated when verification is enabled.
STRUCTURAL_KEYS = (
    "schema_version",
    "source_episode_rel",
    "workspace_episode_rel",
    "object_name",
    "episode",
    "fixed_camera_ids",
    "source_video_directory",
    "pipeline_video_directory",
    "linked_names",
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


def _contract_frame_count(path: Path) -> int:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        indices = sorted(
            int(key.removeprefix("frame_"))
            for key in archive.files
            if key.startswith("frame_")
        )
    if not indices or indices != list(range(len(indices))):
        raise ValueError(f"Frame contract keys are not contiguous from 0: {path}")
    return len(indices)


def _video_frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count <= 0:
        raise ValueError(f"Video reports {count} frames: {path}")
    return count


def _verify_frame_offset(
    episode_dir: Path,
    camera_ids: list[str],
    contract_name: str,
    expected_offset: int,
    probe_cameras: int,
) -> tuple[int, int]:
    """Return (contract_frames, video_frames) and check the expected offset."""
    contract_frames = _contract_frame_count(episode_dir / contract_name)
    probed = camera_ids if probe_cameras <= 0 else camera_ids[:probe_cameras]
    counts = {
        camera_id: _video_frame_count(episode_dir / "vid" / f"{camera_id}.mp4")
        for camera_id in probed
    }
    distinct = set(counts.values())
    if len(distinct) != 1:
        raise ValueError(f"Cameras disagree on frame count: {counts}")
    video_frames = distinct.pop()
    offset = contract_frames - video_frames
    if offset != expected_offset:
        raise ValueError(
            f"Frame offset {offset} (contract {contract_frames} - video "
            f"{video_frames}) differs from --expected-frame-offset={expected_offset}"
        )
    return contract_frames, video_frames


def _manifest_conflict(existing: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    for key in STRUCTURAL_KEYS:
        if existing.get(key) != manifest.get(key):
            return f"{key}: stored={existing.get(key)!r} expected={manifest.get(key)!r}"
    for key in ("contract_frames", "video_frames", "frame_index_offset"):
        stored, current = existing.get(key), manifest.get(key)
        if stored is not None and current is not None and stored != current:
            return f"{key}: stored={stored!r} expected={current!r}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--source-rel", type=Path, default=DEFAULT_SOURCE_REL)
    parser.add_argument("--workspace-rel", type=Path, default=DEFAULT_WORKSPACE_REL)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--expected-cameras", type=int, default=22,
                        help="Required calibrated camera count per episode. 0 disables the check.")
    parser.add_argument("--contract-name", default="object_6d_pose.npz",
                        help="Existing per-episode NPZ that defines the frame contract.")
    parser.add_argument("--expected-frame-offset", type=int, default=3,
                        help="Required contract_frames - video_frames for every episode.")
    verify_group = parser.add_mutually_exclusive_group()
    verify_group.add_argument("--verify-frame-offset", dest="verify_frame_offset",
                              action="store_true", default=True,
                              help="Check the frame offset per episode (default).")
    verify_group.add_argument("--no-verify-frame-offset", dest="verify_frame_offset",
                              action="store_false",
                              help="Skip video probing; leaves the offset unrecorded.")
    parser.add_argument("--frame-offset-probe-cameras", type=int, default=1,
                        help="Videos to probe per episode. 0 probes every calibrated camera.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional JSON summary written even in dry-run mode.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    for value in (args.source_rel, args.workspace_rel):
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("Relative roots must not be absolute or contain '..'")
    if args.frame_offset_probe_cameras < 0:
        raise ValueError("--frame-offset-probe-cameras must be non-negative")

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
    offsets: dict[int, int] = {}
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

            contract_frames: int | None = None
            video_frames: int | None = None
            if args.verify_frame_offset:
                try:
                    contract_frames, video_frames = _verify_frame_offset(
                        episode_dir, camera_ids, args.contract_name,
                        args.expected_frame_offset, args.frame_offset_probe_cameras,
                    )
                except ValueError as exc:
                    raise ValueError(f"{tag}: {exc}") from exc
                offsets[contract_frames - video_frames] = (
                    offsets.get(contract_frames - video_frames, 0) + 1
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
                "contract_frames": contract_frames,
                "video_frames": video_frames,
                "frame_index_offset": (
                    None if contract_frames is None or video_frames is None
                    else contract_frames - video_frames
                ),
            }
            manifest_path = destination / "source_episode.json"
            if manifest_path.exists():
                stored = _read_json(manifest_path)
                conflict = _manifest_conflict(stored, manifest)
                if conflict is not None:
                    raise ValueError(f"Workspace manifest mismatch: {manifest_path} ({conflict})")
                # A run without verification must not erase an offset an earlier
                # run already established.
                for key in ("contract_frames", "video_frames", "frame_index_offset"):
                    if manifest[key] is None and stored.get(key) is not None:
                        manifest[key] = stored[key]
                if args.write and manifest != stored:
                    _atomic_json(manifest_path, manifest)
            elif args.write:
                _atomic_json(manifest_path, manifest)
            planned += 1

    mode = "write" if args.write else "dry-run"
    print(
        f"mode={mode} source={source} workspace={workspace} "
        f"episodes={planned} fixed_video_links={video_links} "
        f"frame_offsets={offsets or 'unverified'}",
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
            "expected_frame_offset": args.expected_frame_offset,
            "verify_frame_offset": args.verify_frame_offset,
            "frame_offset_probe_cameras": args.frame_offset_probe_cameras,
            "observed_frame_offsets": {str(k): v for k, v in sorted(offsets.items())},
        })
        print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
