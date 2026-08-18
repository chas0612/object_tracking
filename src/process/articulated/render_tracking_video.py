#!/usr/bin/env python3
"""Render a 2x2 articulated-mesh reprojection video from GoTrack world poses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import load_articulation, load_cameras  # noqa: E402
from render_overlay import mesh_overlay  # noqa: E402


DEFAULT_VIEWS = ["25452066", "22641023", "23263775", "23028333"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="GoTrack output root containing <object>/world_pose_records.json")
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", nargs=4, default=DEFAULT_VIEWS)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="Render only the first N valid poses; -1 renders all.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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

    articulation = load_articulation(args.object)
    all_cameras = load_cameras(args.capture_dir)
    missing = sorted(set(args.views) - set(all_cameras))
    if missing:
        raise KeyError(f"Missing calibrated views: {missing}")
    cameras = {cid: all_cameras[cid].scaled(args.scale) for cid in args.views}

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (tile_w * 2, tile_h * 2),
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
                panel = mesh_overlay(image, cameras[camera_id], articulation,
                                     pose, theta, crop=False)
                if panel.shape[1] != tile_w or panel.shape[0] != tile_h:
                    panel = cv2.resize(panel, (tile_w, tile_h))
                cv2.putText(panel, f"{camera_id}  f{frame_index}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                panels.append(panel)
            writer.write(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
    finally:
        writer.release()
        for capture in captures.values():
            capture.release()

    print(f"wrote {len(records)} frames to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
