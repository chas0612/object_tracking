"""In-memory camera undistortion for OpenCV video capture streams."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class UndistortingCapture:
    """A small ``cv2.VideoCapture`` proxy that remaps every decoded frame."""

    def __init__(self, capture: Any, calibration: dict[str, Any], *, fixed_point: bool = True):
        self._capture = capture
        source_k = np.asarray(calibration["original_intrinsics"], dtype=np.float64).reshape(3, 3)
        target_k = np.asarray(calibration["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(calibration.get("dist_params", []), dtype=np.float64).reshape(-1)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions for inline undistort: {width}x{height}")
        map_x, map_y = cv2.initUndistortRectifyMap(
            source_k, distortion, None, target_k, (width, height), cv2.CV_32FC1,
        )
        if fixed_point:
            map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        self._map_x = map_x
        self._map_y = map_y

    def read(self):
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return ok, frame
        return True, cv2.remap(
            frame, self._map_x, self._map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )

    def __getattr__(self, name: str):
        return getattr(self._capture, name)


def wrap_capture_dict(
    captures: dict[str, dict[str, Any]], input_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Wrap GoTrack captures when the staged input opts into inline remapping."""
    root = Path(input_root)
    config_path = root / "inline_undistort.json"
    if not config_path.is_file():
        return captures
    config = json.loads(config_path.read_text(encoding="utf-8"))
    intrinsics = json.loads(
        (root / "cam_param" / "intrinsics.json").read_text(encoding="utf-8")
    )
    for camera_id, payload in captures.items():
        record = intrinsics.get(camera_id)
        if not isinstance(record, dict):
            raise KeyError(f"Missing inline-undistort calibration for {camera_id}")
        payload["video_cap"] = UndistortingCapture(
            payload["video_cap"], record,
            fixed_point=bool(config.get("fixed_point_map", True)),
        )
    return captures
