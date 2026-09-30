from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autodex.tracking.latest_gotrack import CaptureReader, frame_indices, world_record, write_json
from src.process.gotrack_capture import _select_cameras


def test_reader_reverse_order_and_scaled_undistorted_intrinsics(tmp_path):
    (tmp_path / "cam_param").mkdir()
    (tmp_path / "videos").mkdir()
    K = [[55, 0, 31], [0, 54, 23], [0, 0, 1]]
    calibration = {"a": {"original_intrinsics": K, "intrinsics_undistort": K,
                          "dist_params": [0, 0, 0, 0]}}
    (tmp_path / "cam_param/intrinsics.json").write_text(json.dumps(calibration))
    (tmp_path / "cam_param/extrinsics.json").write_text(json.dumps({"a": np.eye(4).tolist()}))
    (tmp_path / "inline_undistort.json").write_text("{}")
    writer = cv2.VideoWriter(str(tmp_path / "videos/a.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(12):
        writer.write(np.full((48, 64, 3), 15 * i, dtype=np.uint8))
    writer.release()
    reader = CaptureReader(tmp_path, ["a"], .5)
    try:
        assert reader.total == 12
        np.testing.assert_allclose(reader.K[0], [[27.5, 0, 15.5], [0, 27, 11.5], [0, 0, 1]])
        reverse = [reader.read(i, reverse=True)[0] for i in range(11, -1, -1)]
        forward = [reader.read(i)[0] for i in range(12)]
        assert all(np.array_equal(a, b) for a, b in zip(reverse, reversed(forward)))
        assert forward[0].shape == (24, 32, 3)
    finally:
        reader.close()


def test_seed_window_uses_original_frame_numbers():
    assert list(frame_indices(413, 30, -1, "forward", 35)) == [30, 31, 32, 33, 34]
    assert list(frame_indices(413, 28, 30, "reverse", -1)) == [30, 29, 28]
    with pytest.raises(ValueError):
        frame_indices(413, 30, -1, "forward", 30)


def test_fallback_is_exported_honestly_and_json_has_no_nan(tmp_path):
    result = SimpleNamespace(status="fallback_prior", fusion_backend="infoform_gpu",
                             fit_status="no_observations", fit_score=0, score=0,
                             health={"residual": np.nan}, components_ms={})
    row = world_record(30, 30, np.eye(4), result, 35)
    write_json(tmp_path / "records.json", [row])
    out = json.loads((tmp_path / "records.json").read_text())[0]
    assert out["status"] == "fallback_prior"
    assert out["frame_index"] == 30 and out["time_sec"] == 1
    assert out["health"]["residual"] is None


def test_fps_does_not_duplicate_cameras_at_same_position(tmp_path):
    path = tmp_path / "extrinsics.json"
    path.write_text(json.dumps({c: np.eye(4).tolist() for c in "abcd"}))
    assert len(set(_select_cameras(path, list("abcd"), 4))) == 4
