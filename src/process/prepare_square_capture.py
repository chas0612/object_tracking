#!/usr/bin/env python3
"""Create a calibrated center-square capture without modifying its source.

All selected views are cropped (never stretched) to one common square resolution.
The crop offset is subtracted from the principal point and extrinsics are copied
unchanged.  This is useful for multiview consumers whose bitmap batching requires
identical decoded frame shapes even though the calibrated cameras mix portrait and
landscape sensors.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def center_square_window(width: int, height: int, side: int) -> tuple[int, int, int, int]:
    if side <= 0 or side > min(width, height):
        raise ValueError(f"square side {side} does not fit {width}x{height}")
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    return x0, y0, side, side


def crop_intrinsic(matrix: list[list[float]], x0: int, y0: int) -> list[list[float]]:
    K = np.asarray(matrix, dtype=np.float64).reshape(3, 3).copy()
    K[0, 2] -= float(x0)
    K[1, 2] -= float(y0)
    return K.tolist()


def video_metadata(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(cap.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        cap.release()


def crop_video(source: Path, destination: Path, window: tuple[int, int, int, int]) -> dict[str, object]:
    width, height, expected_frames, fps = video_metadata(source)
    x0, y0, crop_width, crop_height = window
    partial = destination.with_suffix(".partial.avi")
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(partial), cv2.VideoWriter_fourcc(*"MJPG"), fps, (crop_width, crop_height))
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"cannot open crop reader/writer for {source}")
    frames = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if image.shape[:2] != (height, width):
                raise RuntimeError(
                    f"{source} frame {frames}: decoded {image.shape[1]}x{image.shape[0]}, "
                    f"expected {width}x{height}")
            writer.write(image[y0:y0 + crop_height, x0:x0 + crop_width])
            frames += 1
    finally:
        cap.release()
        writer.release()
    if frames != expected_frames:
        raise RuntimeError(f"{source}: wrote {frames}/{expected_frames} frames")
    os.replace(partial, destination)
    return {"frames": frames, "fps": fps, "source_width": width, "source_height": height}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-ids", nargs="+", required=True)
    parser.add_argument("--side", type=int, default=None,
                        help="Output side in pixels; default is the largest common square.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    source = args.capture_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to modify existing output: {output}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    intrinsics = json.loads((source / "cam_param/intrinsics.json").read_text())
    extrinsics = json.loads((source / "cam_param/extrinsics.json").read_text())
    camera_ids = list(dict.fromkeys(args.camera_ids))
    missing = [cid for cid in camera_ids if cid not in intrinsics or cid not in extrinsics
               or not (source / f"undistorted_video/{cid}.avi").is_file()]
    if missing:
        raise ValueError(f"missing video or calibration for cameras: {missing}")

    metadata = {
        cid: video_metadata(source / f"undistorted_video/{cid}.avi")
        for cid in camera_ids
    }
    side = args.side or min(min(width, height) for width, height, _, _ in metadata.values())
    windows: dict[str, tuple[int, int, int, int]] = {}
    cropped_intrinsics: dict[str, dict[str, object]] = {}
    for cid in camera_ids:
        width, height, _, _ = metadata[cid]
        entry = dict(intrinsics[cid])
        if int(entry["width"]) != width or int(entry["height"]) != height:
            raise ValueError(f"camera {cid}: video and calibration dimensions disagree")
        window = center_square_window(width, height, side)
        windows[cid] = window
        x0, y0, _, _ = window
        for key in ("original_intrinsics", "intrinsics_undistort"):
            entry[key] = crop_intrinsic(entry[key], x0, y0)
        entry["width"] = side
        entry["height"] = side
        entry["dist_height"] = side
        cropped_intrinsics[cid] = entry

    (output / "cam_param").mkdir(parents=True)
    (output / "undistorted_video").mkdir()
    (output / "cam_param/intrinsics.json").write_text(
        json.dumps(cropped_intrinsics, indent=2) + "\n", encoding="utf-8")
    (output / "cam_param/extrinsics.json").write_text(
        json.dumps({cid: extrinsics[cid] for cid in camera_ids}, indent=2) + "\n",
        encoding="utf-8")

    results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(camera_ids))) as pool:
        futures = {
            pool.submit(
                crop_video,
                source / f"undistorted_video/{cid}.avi",
                output / f"undistorted_video/{cid}.avi",
                windows[cid],
            ): cid for cid in camera_ids
        }
        for future in as_completed(futures):
            cid = futures[future]
            results[cid] = future.result()
            print(f"[crop] camera {cid}: {results[cid]['frames']} frames", flush=True)

    manifest = {
        "source_capture": str(source),
        "camera_ids": camera_ids,
        "output_resolution": [side, side],
        "crop_windows_xywh": {cid: list(windows[cid]) for cid in camera_ids},
        "videos": results,
        "calibration_update": "principal point minus crop origin; extrinsics unchanged",
    }
    (output / "square_crop_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] calibrated {side}x{side} capture: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
