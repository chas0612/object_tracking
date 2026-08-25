#!/usr/bin/env python3
"""Render a 2x2 articulated-mesh reprojection video from GoTrack world poses."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import load_articulation, load_articulation_from_joint, load_cameras  # noqa: E402
from render_overlay import mesh_overlay  # noqa: E402


DEFAULT_VIEWS = ["25452066", "22641023", "23263775", "23028333"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="GoTrack output root containing <object>/world_pose_records.json")
    parser.add_argument("--object", required=True)
    parser.add_argument("--joint-json", type=Path, default=None,
                        help="Explicit articulation file for an external mesh dataset.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=DEFAULT_VIEWS)
    parser.add_argument("--dynamic-camera-id", default=None,
                        help="View whose per-frame world-to-camera poses are supplied below.")
    parser.add_argument("--dynamic-extrinsics-npy", type=Path, default=None,
                        help="Nx4x4 (or Nx3x4) world-to-camera trajectory.")
    parser.add_argument("--distortion-npy", type=Path, default=None,
                        help="Optional OpenCV distortion coefficients for the dynamic view.")
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="Render only the first N valid poses; -1 renders all.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.views:
        raise ValueError("--views needs at least one camera")
    dynamic_requested = args.dynamic_camera_id is not None or args.dynamic_extrinsics_npy is not None
    if dynamic_requested and (args.dynamic_camera_id is None or args.dynamic_extrinsics_npy is None):
        raise ValueError("dynamic rendering requires both --dynamic-camera-id and --dynamic-extrinsics-npy")
    if args.dynamic_camera_id is not None and args.dynamic_camera_id not in args.views:
        raise ValueError("--dynamic-camera-id must also be listed in --views")
    if args.distortion_npy is not None and not dynamic_requested:
        raise ValueError("--distortion-npy requires a dynamic camera")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    records_path = args.run_dir / args.object / "world_pose_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records = [row for row in records
               if row.get("status") == "ok" and row.get("pose_world") is not None]
    if not records:
        raise ValueError(f"No valid poses in {records_path}")
    if args.max_frames >= 0:
        records = records[:args.max_frames]

    articulation = (load_articulation_from_joint(args.joint_json)
                    if args.joint_json is not None else load_articulation(args.object))
    all_cameras = load_cameras(args.capture_dir)
    missing = sorted(set(args.views) - set(all_cameras))
    if missing:
        raise KeyError(f"Missing calibrated views: {missing}")
    cameras = {cid: all_cameras[cid].scaled(args.scale) for cid in args.views}
    dynamic_extrinsics = None
    distortion = None
    if dynamic_requested:
        dynamic_extrinsics = np.asarray(
            np.load(args.dynamic_extrinsics_npy.expanduser().resolve()), dtype=np.float64)
        if dynamic_extrinsics.ndim != 3 or dynamic_extrinsics.shape[1:] not in ((3, 4), (4, 4)):
            raise ValueError(
                f"{args.dynamic_extrinsics_npy}: expected Nx3x4 or Nx4x4, "
                f"got {dynamic_extrinsics.shape}")
        if dynamic_extrinsics.shape[1:] == (3, 4):
            bottom = np.broadcast_to([0.0, 0.0, 0.0, 1.0],
                                     (len(dynamic_extrinsics), 1, 4))
            dynamic_extrinsics = np.concatenate([dynamic_extrinsics, bottom], axis=1)
        if not np.isfinite(dynamic_extrinsics).all():
            raise ValueError(f"{args.dynamic_extrinsics_npy}: non-finite camera poses")
        if max(int(row["frame_index"]) for row in records) >= len(dynamic_extrinsics):
            raise ValueError("dynamic camera trajectory is shorter than the tracking records")
        if args.distortion_npy is not None:
            distortion = np.asarray(
                np.load(args.distortion_npy.expanduser().resolve()), dtype=np.float64).reshape(-1)
            if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
                raise ValueError(f"{args.distortion_npy}: invalid OpenCV distortion vector")

    video_dir = args.capture_dir / "undistorted_video"
    captures = {cid: cv2.VideoCapture(str(video_dir / f"{cid}.avi")) for cid in args.views}
    unopened = [cid for cid, capture in captures.items() if not capture.isOpened()]
    if unopened:
        raise FileNotFoundError(f"Cannot open undistorted videos for {unopened}")
    # Avoid an expensive decoder seek for the common case of consecutive records.
    # Seeking every AVI on every frame made a full 1,851-frame grid take roughly
    # twenty minutes even though OpenCV had already advanced to the desired frame.
    next_frames = {cid: 0 for cid in args.views}

    tile_w = max(camera.width for camera in cameras.values())
    tile_h = max(camera.height for camera in cameras.values())
    columns = 1 if len(args.views) == 1 else int(math.ceil(math.sqrt(len(args.views))))
    rows = int(math.ceil(len(args.views) / columns))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (tile_w * columns, tile_h * rows),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {args.output}")

    try:
        for row in tqdm(records, desc="articulated reprojection"):
            frame_index = int(row["frame_index"])
            pose = np.asarray(row["pose_world"], dtype=np.float64)
            theta = float(row.get("joint_value", row.get("theta_rad", 0.0)))
            panels = []
            for camera_id in args.views:
                capture = captures[camera_id]
                if next_frames[camera_id] != frame_index:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot read {camera_id} frame {frame_index}")
                next_frames[camera_id] = frame_index + 1
                camera = cameras[camera_id]
                if camera_id == args.dynamic_camera_id:
                    camera = type(camera)(
                        camera_id=camera.camera_id,
                        K=camera.K,
                        extrinsic=dynamic_extrinsics[frame_index],
                        width=camera.width,
                        height=camera.height,
                    )
                    if distortion is not None:
                        image = cv2.resize(image, (camera.width, camera.height),
                                           interpolation=cv2.INTER_AREA)
                        image = cv2.undistort(image, camera.K, distortion, None, camera.K)
                panel = mesh_overlay(image, camera, articulation,
                                     pose, theta, crop=False)
                if panel.shape[1] != tile_w or panel.shape[0] != tile_h:
                    panel = cv2.resize(panel, (tile_w, tile_h))
                cv2.putText(panel, f"{camera_id}  f{frame_index}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                panels.append(panel)
            blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            panels.extend(blank.copy() for _ in range(columns * rows - len(panels)))
            grid_rows = [np.hstack(panels[start:start + columns])
                         for start in range(0, len(panels), columns)]
            writer.write(np.vstack(grid_rows))
    finally:
        writer.release()
        for capture in captures.values():
            capture.release()

    print(f"wrote {len(records)} frames to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
