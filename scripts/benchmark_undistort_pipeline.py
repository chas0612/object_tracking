#!/usr/bin/env python3
"""Measure decode, remap, and encode costs of the capture undistort path.

The output video is temporary and may be placed on local storage to avoid
adding NAS writes.  This tool never modifies the source capture.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


def _run(
    video: Path, calibration: dict, output: Path | None, max_frames: int,
    fixed_point_map: bool,
) -> dict:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_k = np.asarray(calibration["original_intrinsics"], dtype=np.float64).reshape(3, 3)
    target_k = np.asarray(calibration["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(calibration["dist_params"], dtype=np.float64).reshape(-1)
    t0 = time.perf_counter()
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_k, distortion, None, target_k, (width, height), cv2.CV_32FC1,
    )
    if fixed_point_map:
        map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    map_sec = time.perf_counter() - t0
    writer = None
    if output is not None:
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height), True,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create {output}")

    decode_sec = remap_sec = encode_sec = 0.0
    frames = 0
    while frames < max_frames:
        t0 = time.perf_counter()
        ok, frame = capture.read()
        decode_sec += time.perf_counter() - t0
        if not ok:
            break
        t0 = time.perf_counter()
        remapped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        remap_sec += time.perf_counter() - t0
        if writer is not None:
            t0 = time.perf_counter()
            writer.write(remapped)
            encode_sec += time.perf_counter() - t0
        frames += 1
    capture.release()
    if writer is not None:
        writer.release()
    measured = decode_sec + remap_sec + encode_sec
    return {
        "video": str(video), "frames": frames, "width": width, "height": height,
        "fixed_point_map": fixed_point_map,
        "map_sec": map_sec, "decode_read_sec": decode_sec,
        "remap_sec": remap_sec, "encode_write_sec": encode_sec,
        "measured_loop_sec": measured,
        "fps": frames / measured if measured else None,
        "fractions": {
            "decode_read": decode_sec / measured if measured else None,
            "remap": remap_sec / measured if measured else None,
            "encode_write": encode_sec / measured if measured else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--intrinsics-json", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--local-copy", action="store_true")
    parser.add_argument("--no-encode", action="store_true")
    parser.add_argument("--fixed-point-map", action="store_true")
    args = parser.parse_args()
    if args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    video = args.video.expanduser().resolve()
    calibration = json.loads(args.intrinsics_json.expanduser().read_text(encoding="utf-8"))
    if args.camera_id not in calibration:
        raise KeyError(args.camera_id)

    with tempfile.TemporaryDirectory(prefix="undistort_bench_") as temporary:
        temporary_root = Path(temporary)
        measured_video = video
        copy_sec = 0.0
        if args.local_copy:
            measured_video = temporary_root / video.name
            t0 = time.perf_counter()
            shutil.copyfile(video, measured_video)
            copy_sec = time.perf_counter() - t0
        output = None if args.no_encode else temporary_root / "output.avi"
        result = _run(
            measured_video, calibration[args.camera_id], output, args.max_frames,
            args.fixed_point_map,
        )
        result.update({
            "source_video": str(video), "local_copy": bool(args.local_copy),
            "copy_sec": copy_sec, "source_bytes": video.stat().st_size,
        })
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
