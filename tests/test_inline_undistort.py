from __future__ import annotations

import cv2
import json
import numpy as np

from autodex.tracking.inline_undistort import UndistortingCapture
from src.process.gotrack_capture import _merge_bidirectional_records


class _FakeCapture:
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frame.shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frame.shape[0])
        return 0.0

    def read(self):
        return True, self.frame.copy()


def test_capture_matches_direct_fixed_point_remap() -> None:
    height, width = 48, 64
    frame = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    source_k = np.array([[55.0, 0.0, 31.0], [0.0, 54.0, 23.0], [0.0, 0.0, 1.0]])
    target_k = np.array([[53.0, 0.0, 31.0], [0.0, 52.0, 23.0], [0.0, 0.0, 1.0]])
    distortion = np.array([0.08, -0.03, 0.001, -0.002, 0.0])
    calibration = {
        "original_intrinsics": source_k.reshape(-1).tolist(),
        "intrinsics_undistort": target_k.reshape(-1).tolist(),
        "dist_params": distortion.tolist(),
    }
    wrapped = UndistortingCapture(_FakeCapture(frame), calibration, fixed_point=True)
    ok, actual = wrapped.read()
    float_x, float_y = cv2.initUndistortRectifyMap(
        source_k, distortion, None, target_k, (width, height), cv2.CV_32FC1,
    )
    map_x, map_y = cv2.convertMaps(float_x, float_y, cv2.CV_16SC2)
    expected = cv2.remap(
        frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    assert ok
    assert np.array_equal(actual, expected)


def test_direct_reverse_records_keep_original_frame_indices(tmp_path) -> None:
    pose = np.eye(4).tolist()
    forward = tmp_path / "forward.json"
    backward = tmp_path / "backward.json"
    merged = tmp_path / "merged.json"
    forward.write_text(json.dumps([
        {"frame_index": 30, "pose_world": pose},
        {"frame_index": 31, "pose_world": pose},
    ]), encoding="utf-8")
    backward.write_text(json.dumps([
        {"frame_index": 30, "pose_world": pose},
        {"frame_index": 29, "pose_world": pose},
        {"frame_index": 28, "pose_world": pose},
    ]), encoding="utf-8")
    stats = _merge_bidirectional_records(
        forward, backward, merged, 30, backward_uses_original_indices=True,
    )
    records = json.loads(merged.read_text(encoding="utf-8"))
    assert [row["frame_index"] for row in records] == [28, 29, 30, 31]
    assert stats["backward_records_used"] == 2
