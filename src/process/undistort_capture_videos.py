#!/usr/bin/env python3
"""Create calibration-consistent undistorted videos for a capture archive.

The output is ``<capture-dir>/undistorted_video/``.  Source ``videos/`` and
all pre-existing pipeline output are read-only.  Crucially, the remap uses the
provided ``intrinsics_undistort`` matrix directly, rather than recomputing a
new camera matrix with OpenCV.  The resulting pixels therefore match the K
matrix used by FoundPose and GoTrack. Source videos may be AVI or MP4; output
is always AVI because the downstream GoTrack wrapper consumes ``*.avi``.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def _complete_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        return cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        cap.release()


def _process_video(
    source: Path,
    output_dir: Path,
    record: dict,
    overwrite: bool,
    fixed_point_map: bool,
) -> None:
    camera_id = source.stem
    output = output_dir / f"{camera_id}.avi"
    if _complete_video(output) and not overwrite:
        print(f"[skip] {output.name}", flush=True)
        return
    required = {"original_intrinsics", "intrinsics_undistort", "dist_params"}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError(f"Missing calibration fields for camera {camera_id}: need {sorted(required)}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    K_source = np.asarray(record["original_intrinsics"], dtype=np.float64).reshape(3, 3)
    K_output = np.asarray(record["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(record["dist_params"], dtype=np.float64).reshape(-1)
    map_x, map_y = cv2.initUndistortRectifyMap(
        K_source, distortion, None, K_output, (width, height), cv2.CV_32FC1,
    )
    if fixed_point_map:
        map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    temporary = output.with_name(output.stem + ".partial" + output.suffix)
    if temporary.exists():
        temporary.unlink()
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height), True,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create {temporary}")
    print(f"[run] {source.name}: {n_frames} frames", flush=True)
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR))
            written += 1
    finally:
        cap.release()
        writer.release()
    if written != n_frames or not _complete_video(temporary):
        raise RuntimeError(f"Incomplete output for {source.name}: wrote {written}/{n_frames} frames")
    temporary.replace(output)
    print(f"[done] {output.name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", default=None,
                        help="Default: <capture-dir>/undistorted_video/")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing generated video. Never changes videos/.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Camera videos processed concurrently. Default 1 preserves legacy behavior.",
    )
    parser.add_argument(
        "--fixed-point-map", action="store_true",
        help="Use OpenCV's equivalent compact remap tables (faster on CPU).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    source_dir = capture_dir / "videos"
    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
                  capture_dir / "undistorted_video")
    intrinsics_path = capture_dir / "cam_param" / "intrinsics.json"
    if not source_dir.is_dir() or not intrinsics_path.is_file():
        raise FileNotFoundError("Expected videos/ and cam_param/intrinsics.json under --capture-dir")
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "undistort_manifest.json"
    calibration = json.loads(intrinsics_path.read_text(encoding="utf-8"))

    videos_by_camera: dict[str, Path] = {}
    for suffix in ("*.avi", "*.mp4"):
        for source in sorted(source_dir.glob(suffix)):
            if source.stem in videos_by_camera:
                raise ValueError(
                    f"Multiple source videos share camera ID {source.stem}: "
                    f"{videos_by_camera[source.stem]} and {source}"
                )
            videos_by_camera[source.stem] = source
    # Only fixed cameras with explicit calibration are valid FoundPose/GoTrack
    # inputs. Human captures can also contain ego videos whose calibration is
    # stored separately or absent.
    videos = [
        source for camera_id, source in sorted(videos_by_camera.items())
        if camera_id in calibration
    ]
    if not videos:
        raise FileNotFoundError(
            f"No calibrated .avi/.mp4 videos found under {source_dir}"
        )
    print(f"[undistort] {len(videos)} source videos -> {output_dir}", flush=True)
    if args.workers == 1:
        for source in videos:
            _process_video(
                source, output_dir, calibration[source.stem], args.overwrite,
                args.fixed_point_map,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _process_video, source, output_dir,
                    calibration[source.stem], args.overwrite, args.fixed_point_map,
                )
                for source in videos
            ]
            # Resolve in input order for deterministic exception propagation.
            for future in futures:
                future.result()
    manifest_path.write_text(json.dumps({
        "source_dir": str(source_dir),
        "calibration": str(intrinsics_path),
        "new_intrinsics_field": "intrinsics_undistort",
        "camera_workers": args.workers,
        "fixed_point_map": args.fixed_point_map,
        "files": [
            {"source": path.name, "output": f"{path.stem}.avi"}
            for path in videos
        ],
        "excluded_uncalibrated_sources": sorted(
            path.name for camera_id, path in videos_by_camera.items()
            if camera_id not in calibration
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
