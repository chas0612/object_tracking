from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import export_latest_gotrack_poses as export
from src.process.gotrack_capture import _stage_input, _video_paths_by_stem
from src.process.view_gotrack_viser import _robot_configuration


def test_container_selection_and_inline_aliases(tmp_path):
    capture = tmp_path / "capture"
    (capture / "cam_param").mkdir(parents=True)
    videos = capture / "videos"
    videos.mkdir()
    for name in ("a.mp4", "a.avi", "b.mp4", "c.mov", "notes.txt"):
        (videos / name).touch()
    paths = _video_paths_by_stem(videos)
    assert {key: value.name for key, value in paths.items()} == {
        "a": "a.avi", "b": "b.mp4", "c": "c.mov",
    }
    stage = tmp_path / "stage"
    _stage_input(capture, videos, stage, inline_undistort=True)
    assert (stage / "videos/b.avi").resolve() == (videos / "b.mp4").resolve()
    assert (stage / "inline_undistort.json").is_file()


def test_robotiq_configuration_names_only_independent_joints():
    cfg = _robot_configuration(np.arange(7, dtype=float))
    assert cfg == {**{f"joint{i}": float(i - 1) for i in range(1, 7)},
                   "robotiq_85_left_knuckle_joint": 6.0}
    inspire = np.arange(12, dtype=float)
    assert _robot_configuration(inspire) is inspire


def test_scheduled_only_excludes_manifest_only_episodes(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    target = shared / "target"
    for obj in ("apple", "banana"):
        episode = target / obj / "0"
        episode.mkdir(parents=True)
        np.savez(episode / "object_6d_pose.npz", frame_0=np.eye(4))
    schedule = shared / "schedule"
    (schedule / "tasks").mkdir(parents=True)
    (schedule / "manifest.json").write_text("{}")
    (schedule / "tasks/apple_0.json").write_text(json.dumps({
        "object_name": "apple", "episode": "0", "status": "completed", "attempt_dir": "attempt",
    }))
    record = shared / "attempt/gotrack_tracking/gotrack_output/apple/world_pose_records.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps([{"frame_index": 0, "pose_world": np.eye(4).tolist()}]))
    manifest = shared / "latest.json"
    manifest.write_text(json.dumps({"episodes": [{
        "object_name": "banana", "episode": "0", "latest_status": "completed", "record": str(record),
    }]}))
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["export", "--shared-root", str(shared),
        "--target-root", str(target), "--latest-manifest", str(manifest),
        "--schedule-dir", str(schedule), "--scheduled-only", "--report", str(report)])
    assert export.main() == 0
    result = json.loads(report.read_text())
    assert [(row["object_name"], row["episode"]) for row in result["episodes"]] == [("apple", "0")]
