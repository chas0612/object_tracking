#!/usr/bin/env python3
"""Audit, preview, and export successful CORL rebuttal grasp snapshots.

This is deliberately CPU-only.  It treats the camera timestamp stream as the
15 Hz clock and maps it onto the encoded 30 Hz video using each episode's
observed endpoint ratio.  The default ``audit`` mode is read-only.

Examples:
  python scripts/prepare_corl_grasp_snapshots.py audit --manifest-out /tmp/audit.json
  python scripts/prepare_corl_grasp_snapshots.py contacts --output-dir /tmp/grasp_contacts
  python scripts/prepare_corl_grasp_snapshots.py export --selections selections.json

``selections.json`` is a JSON object mapping task IDs printed by ``audit`` to
final encoded video-frame indices.  Export adds ``raw/images/*.png`` and a
separate ``grasp_snapshot/robot_state/`` archive to each selected episode.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PEOPLE = ("seongho", "taeyoon", "kanghyun")
EXPECTED_HUMAN_EPISODES = set(range(5))
PREFERRED_CAMERAS = ("25452066", "22641023", "23263775", "23028333")
OBJECT_ALIASES = {
    "knife_sharper": "knife_sharpener",
    "knife_sharpener": "knife_sharpener",
    "mug_holder": "mug_holder",
    "organizer_beige": "organizer_beige",
    "plastic_elephant_jug": "plastic_elephant_jug",
    "plastic_elephanant_jug": "plastic_elephant_jug",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_float(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)


def _video_metadata(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frames < 1 or fps <= 0 or width < 1 or height < 1:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return frames, fps, width, height


def _discover(root: Path, closure_fraction: float,
              candidate_time_offset_sec: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    successes: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str, int]] = Counter()
    trials = 0
    for person in PEOPLE:
        person_dir = root / person
        if not person_dir.is_dir():
            raise FileNotFoundError(person_dir)
        for result_path in sorted(person_dir.glob("*/*/grasp_result.json")):
            trials += 1
            episode_dir = result_path.parent
            source_object = episode_dir.parent.name
            if source_object not in OBJECT_ALIASES:
                raise KeyError(f"Unknown object alias: {source_object}")
            object_name = OBJECT_ALIASES[source_object]
            paired_path = episode_dir / "paired_human_episode.json"
            result = _read_json(result_path)
            paired = _read_json(paired_path)
            human_episode = int(paired["paired human episode"])
            if not bool(result.get("grasp_success", False)):
                continue
            counts[(person, object_name, human_episode)] += 1

            timestamps = _load_float(episode_dir / "raw/timestamps/timestamp.npy").reshape(-1)
            frame_ids = np.asarray(
                np.load(episode_dir / "raw/timestamps/frame_id.npy", allow_pickle=True)
            ).reshape(-1)
            if len(timestamps) < 2 or len(frame_ids) != len(timestamps):
                raise ValueError(f"Bad camera timestamps: {episode_dir}")
            videos = sorted((episode_dir / "videos").glob("*.avi"))
            if not videos:
                raise FileNotFoundError(f"No AVI videos: {episode_dir}")
            reference = next(
                (episode_dir / "videos" / f"{camera}.avi" for camera in PREFERRED_CAMERAS
                 if (episode_dir / "videos" / f"{camera}.avi").is_file()),
                videos[0],
            )
            frame_count, fps, width, height = _video_metadata(reference)
            ratio = (frame_count - 1) / (len(timestamps) - 1)
            if abs(fps - 30.0) > 0.05 or not 1.95 <= ratio <= 2.05:
                raise ValueError(
                    f"Unexpected video/timestamp rate for {episode_dir}: fps={fps}, ratio={ratio}"
                )

            arm_time = _load_float(episode_dir / "raw/arm/time.npy").reshape(-1)
            arm_action = _load_float(episode_dir / "raw/arm/action.npy")
            hand_time = _load_float(episode_dir / "raw/hand/time.npy").reshape(-1)
            hand_position = _load_float(episode_dir / "raw/hand/position.npy")
            if hand_position.shape[0] != len(hand_time) or not len(hand_time):
                raise ValueError(f"Bad robot trajectory shapes: {episode_dir}")
            # A successful trial can include release and retreat, so the final
            # frame is not the grasp.  Detect the highest commanded wrist pose
            # while the hand is substantially displaced from its initial/open
            # state.  This picks the held-object lift and rejects the open-hand
            # home pose in the common Allegro V5 sequence.
            # Allegro flexion coordinates are positive for closing.  Their
            # sum distinguishes a truly closed grasp from a retreated/open
            # hand more reliably than distance from the initial posture.
            closure = hand_position.sum(axis=1)
            closure_min = float(closure.min())
            closure_max = float(closure.max())
            threshold = closure_min + closure_fraction * (closure_max - closure_min)
            if len(arm_time) and arm_action.shape == (len(arm_time), 4, 4):
                closure_at_arm = np.interp(arm_time, hand_time, closure)
                eligible = np.flatnonzero(closure_at_arm >= threshold)
                if not len(eligible):
                    eligible = np.arange(len(arm_time))
                wrist_z = arm_action[:, 2, 3]
                lift_index = int(eligible[np.argmax(wrist_z[eligible])])
                candidate_time = float(arm_time[lift_index] + candidate_time_offset_sec)
                heuristic = "max_wrist_z_while_hand_closed"
                candidate_wrist_z: float | None = float(wrist_z[lift_index])
                candidate_closure = float(closure_at_arm[lift_index])
            else:
                # A small number of captures have empty arm arrays.  The peak
                # hand displacement is still a useful visual-review seed.
                lift_index = int(np.argmax(closure))
                candidate_time = float(hand_time[lift_index] + candidate_time_offset_sec)
                heuristic = "max_hand_displacement_arm_log_missing"
                candidate_wrist_z = None
                candidate_closure = float(closure[lift_index])
            candidate_time = float(np.clip(candidate_time, timestamps[0], timestamps[-1]))
            timestamp_index = float(np.interp(candidate_time, timestamps, np.arange(len(timestamps))))
            candidate_frame = int(np.clip(round(timestamp_index * ratio), 0, frame_count - 1))
            task_id = f"{person}__{object_name}__human_{human_episode}"
            successes.append({
                "task_id": task_id,
                "person": person,
                "source_object": source_object,
                "object_name": object_name,
                "human_episode": human_episode,
                "robot_episode": int(episode_dir.name),
                "episode_rel": str(episode_dir.relative_to(root)),
                "timestamp_count": int(len(timestamps)),
                "timestamp_start": float(timestamps[0]),
                "timestamp_end": float(timestamps[-1]),
                "reference_camera": reference.stem,
                "video_frame_count": frame_count,
                "video_fps": fps,
                "video_width": width,
                "video_height": height,
                "video_frames_per_timestamp": ratio,
                "candidate_time": candidate_time,
                "candidate_heuristic": heuristic,
                "candidate_signal_index": lift_index,
                "candidate_wrist_z": candidate_wrist_z,
                "candidate_hand_closure": candidate_closure,
                "hand_closure_threshold": threshold,
                "candidate_timestamp_index": timestamp_index,
                "candidate_frame": candidate_frame,
            })

    problems = []
    groups = defaultdict(set)
    for person, object_name, human_episode in counts:
        groups[(person, object_name)].add(human_episode)
    expected_groups = {(person, obj) for person in PEOPLE for obj in set(OBJECT_ALIASES.values())}
    for person, object_name in sorted(expected_groups):
        found = groups.get((person, object_name), set())
        if found != EXPECTED_HUMAN_EPISODES:
            problems.append({
                "person": person, "object": object_name,
                "missing_human_episodes": sorted(EXPECTED_HUMAN_EPISODES - found),
                "unexpected_human_episodes": sorted(found - EXPECTED_HUMAN_EPISODES),
            })
        for human_episode in EXPECTED_HUMAN_EPISODES:
            count = counts[(person, object_name, human_episode)]
            if count != 1:
                problems.append({
                    "person": person, "object": object_name,
                    "human_episode": human_episode, "successful_trials": count,
                })
    task_ids = [task["task_id"] for task in successes]
    if len(task_ids) != len(set(task_ids)):
        problems.append({"duplicate_task_ids": sorted(k for k, v in Counter(task_ids).items() if v > 1)})
    audit = {
        "root": str(root),
        "people": list(PEOPLE),
        "canonical_objects": sorted(set(OBJECT_ALIASES.values())),
        "trials": trials,
        "successful_trials": len(successes),
        "expected_successful_trials": len(PEOPLE) * len(set(OBJECT_ALIASES.values())) * 5,
        "validation": "ok" if not problems else "failed",
        "problems": problems,
    }
    return sorted(successes, key=lambda task: task["task_id"]), audit


def _read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {frame_index}: {path}")
    return frame


def _contact_sheet(root: Path, task: dict[str, Any], output: Path,
                   offsets: list[float], cell_width: int) -> None:
    episode = root / task["episode_rel"]
    available = {path.stem: path for path in (episode / "videos").glob("*.avi")}
    cameras = [camera for camera in PREFERRED_CAMERAS if camera in available]
    if not cameras:
        cameras = sorted(available)[:4]
    fps = float(task["video_fps"])
    base = int(task["candidate_frame"])
    rows = []
    for camera in cameras:
        cells = []
        total, _, width, height = _video_metadata(available[camera])
        cell_height = round(height * cell_width / width)
        for offset in offsets:
            index = int(np.clip(round(base + offset * fps), 0, total - 1))
            frame = cv2.resize(_read_frame(available[camera], index), (cell_width, cell_height))
            cv2.rectangle(frame, (0, 0), (cell_width, 31), (0, 0, 0), -1)
            cv2.putText(
                frame, f"{camera}  f={index}  {offset:+.1f}s", (7, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 255), 1, cv2.LINE_AA,
            )
            cells.append(frame)
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write {output}")


def _nearest(values: np.ndarray, timestamp: float) -> tuple[int, float]:
    index = int(np.abs(values - timestamp).argmin())
    return index, float(values[index] - timestamp)


def _robot_snapshot(episode: Path, timestamp: float,
                    allow_missing_arm: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = episode / "raw"
    arm_time = _load_float(raw / "arm/time.npy").reshape(-1)
    hand_time = _load_float(raw / "hand/time.npy").reshape(-1)
    hand_index, hand_delta = _nearest(hand_time, timestamp)
    arrays = {
        "capture_timestamp": np.asarray(timestamp, dtype=np.float64),
        "hand_sample_index": np.asarray(hand_index, dtype=np.int64),
        "hand_sample_timestamp": np.asarray(hand_time[hand_index], dtype=np.float64),
        "hand_position": _load_float(raw / "hand/position.npy")[hand_index],
        "hand_action": _load_float(raw / "hand/action.npy")[hand_index],
    }
    metadata: dict[str, Any] = {
        "capture_timestamp": timestamp,
        "arm_available": bool(len(arm_time)),
        "hand_sample_index": hand_index,
        "hand_sample_timestamp": float(hand_time[hand_index]),
        "hand_time_delta_sec": hand_delta,
    }
    arm_fields = {
        "arm_position": raw / "arm/position.npy",
        "arm_velocity": raw / "arm/velocity.npy",
        "arm_torque": raw / "arm/torque.npy",
        "arm_action_qpos": raw / "arm/action_qpos.npy",
        "arm_action_ee_transform": raw / "arm/action.npy",
    }
    if len(arm_time):
        arm_index, arm_delta = _nearest(arm_time, timestamp)
        arrays["arm_sample_index"] = np.asarray(arm_index, dtype=np.int64)
        arrays["arm_sample_timestamp"] = np.asarray(arm_time[arm_index], dtype=np.float64)
        for name, path in arm_fields.items():
            values = _load_float(path)
            if values.shape[0] != len(arm_time):
                raise ValueError(f"Arm field/time length mismatch: {path}")
            arrays[name] = values[arm_index]
        metadata.update({
            "arm_sample_index": arm_index,
            "arm_sample_timestamp": float(arm_time[arm_index]),
            "arm_time_delta_sec": arm_delta,
        })
    else:
        if not allow_missing_arm:
            raise RuntimeError(
                f"Arm state arrays are empty for {episode}; pass --allow-missing-arm-state "
                "only if a hand-only snapshot is acceptable"
            )
        arrays["arm_sample_index"] = np.asarray(-1, dtype=np.int64)
        arrays["arm_sample_timestamp"] = np.asarray(np.nan, dtype=np.float64)
        for name in arm_fields:
            arrays[name] = np.empty((0,), dtype=np.float64)
        metadata.update({
            "arm_sample_index": None, "arm_sample_timestamp": None,
            "arm_time_delta_sec": None,
            "warning": "source arm arrays are empty; hand state only",
        })
    metadata["fields"] = {name: list(value.shape) for name, value in arrays.items()}
    return arrays, metadata


def _frame_timestamp(episode: Path, frame_index: int, video_frames: int) -> tuple[float, float]:
    timestamps = _load_float(episode / "raw/timestamps/timestamp.npy").reshape(-1)
    if video_frames < 2:
        raise ValueError("video must contain at least two frames")
    timestamp_index = frame_index * (len(timestamps) - 1) / (video_frames - 1)
    timestamp = float(np.interp(timestamp_index, np.arange(len(timestamps)), timestamps))
    return timestamp, float(timestamp_index)


def _export(root: Path, task: dict[str, Any], frame_index: int, overwrite: bool,
            allow_missing_arm: bool) -> None:
    episode = root / task["episode_rel"]
    videos = sorted((episode / "videos").glob("*.avi"))
    if not videos:
        raise FileNotFoundError(episode / "videos")
    reference = episode / "videos" / f"{task['reference_camera']}.avi"
    video_frames, _, _, _ = _video_metadata(reference)
    if not 0 <= frame_index < video_frames:
        raise IndexError(f"{task['task_id']}: frame {frame_index} outside [0,{video_frames - 1}]")
    image_dir = episode / "raw/images"
    snapshot_dir = episode / "grasp_snapshot"
    state_dir = snapshot_dir / "robot_state"
    targets = [image_dir, snapshot_dir / "selection.json", state_dir / "robot_state.npz"]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing existing snapshot output under {episode}; use --overwrite")
    image_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        total, _, _, _ = _video_metadata(video)
        camera_frame = int(np.clip(round(frame_index * (total - 1) / (video_frames - 1)), 0, total - 1))
        output = image_dir / f"{video.stem}.png"
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        if not cv2.imwrite(str(output), _read_frame(video, camera_frame)):
            raise RuntimeError(f"Could not write {output}")
    timestamp, timestamp_index = _frame_timestamp(episode, frame_index, video_frames)
    arrays, state_metadata = _robot_snapshot(episode, timestamp, allow_missing_arm)
    state_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_dir / "robot_state.npz", **arrays)
    state_metadata.update({
        "source_episode": str(episode),
        "selected_video_frame": frame_index,
        "camera_timestamp_index_float": timestamp_index,
    })
    _atomic_json(state_dir / "metadata.json", state_metadata)
    selection = dict(task)
    selection.update({
        "selected_frame": frame_index,
        "selected_timestamp": timestamp,
        "selected_timestamp_index_float": timestamp_index,
        "raw_images_dir": str(image_dir),
        "robot_state_npz": str(state_dir / "robot_state.npz"),
    })
    _atomic_json(snapshot_dir / "selection.json", selection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "contacts", "selections", "export"))
    parser.add_argument(
        "--root", default=str(Path.home() / "shared_data/capture/corl_rebuttal")
    )
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None,
                        help="Limit contact-sheet generation to these audited task IDs.")
    parser.add_argument("--closure-fraction", type=float, default=0.65)
    parser.add_argument("--candidate-time-offset-sec", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--offsets-sec", nargs="+", type=float, default=[-2.0, -1.0, 0.0, 0.5, 1.0])
    parser.add_argument("--cell-width", type=int, default=384)
    parser.add_argument("--selections", default=None)
    parser.add_argument("--offset-config", default=None)
    parser.add_argument("--selections-out", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing-arm-state", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.closure_fraction <= 1 or args.cell_width < 96:
        raise ValueError("invalid closure fraction or cell width")
    root = Path(args.root).expanduser().resolve()
    tasks, audit = _discover(root, args.closure_fraction, args.candidate_time_offset_sec)
    manifest = {"audit": audit, "tasks": tasks}
    print(json.dumps(audit, indent=2))
    if audit["validation"] != "ok":
        return 1
    if args.manifest_out:
        _atomic_json(Path(args.manifest_out).expanduser().resolve(), manifest)
        print(f"[wrote] {args.manifest_out}")
    if args.mode == "audit":
        for task in tasks:
            print(
                f"{task['task_id']} robot_ep={task['robot_episode']} "
                f"candidate_frame={task['candidate_frame']}"
            )
        return 0
    if args.mode == "contacts":
        if not args.output_dir:
            raise ValueError("contacts requires --output-dir")
        if args.task_ids:
            requested = set(args.task_ids)
            known = {task["task_id"] for task in tasks}
            unknown = sorted(requested - known)
            if unknown:
                raise KeyError(f"Unknown task IDs: {unknown}")
            contact_tasks = [task for task in tasks if task["task_id"] in requested]
        else:
            contact_tasks = tasks
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, task in enumerate(contact_tasks, 1):
            output = output_dir / f"{task['task_id']}.jpg"
            if output.exists() and not args.overwrite:
                print(f"[skip] {output}")
                continue
            _contact_sheet(root, task, output, args.offsets_sec, args.cell_width)
            print(f"[{index}/{len(contact_tasks)}] {output}", flush=True)
        _atomic_json(output_dir / "candidate_manifest.json", manifest)
        _atomic_json(
            output_dir / "selections_template.json",
            {task["task_id"]: task["candidate_frame"] for task in tasks},
        )
        return 0
    if args.mode == "selections":
        if not args.offset_config or not args.selections_out:
            raise ValueError("selections requires --offset-config and --selections-out")
        config = _read_json(Path(args.offset_config).expanduser().resolve())
        default_offset = float(config["default_offset_sec"])
        overrides = config.get("overrides", {})
        time_overrides = config.get("time_sec_overrides", {})
        if not isinstance(overrides, dict) or not isinstance(time_overrides, dict):
            raise ValueError("offset/time overrides must be JSON objects")
        known = {task["task_id"] for task in tasks}
        unknown = sorted((set(overrides) | set(time_overrides)) - known)
        if unknown:
            raise KeyError(f"Unknown task IDs in offset config: {unknown}")
        selections: dict[str, int | None] = {}
        decisions: list[dict[str, Any]] = []
        for task in tasks:
            if task["task_id"] in time_overrides:
                video_time_sec = float(time_overrides[task["task_id"]])
                frame = int(np.clip(
                    round(video_time_sec * task["video_fps"]),
                    0, task["video_frame_count"] - 1,
                ))
                offset = (frame - task["candidate_frame"]) / task["video_fps"]
                status = "selected"
                source = "absolute_video_time"
            else:
                offset = overrides.get(task["task_id"], default_offset)
                source = "candidate_offset"
            if source == "candidate_offset" and offset is None:
                frame = None
                video_time_sec = None
                status = "pending_manual_review"
            elif source == "candidate_offset":
                offset = float(offset)
                frame = int(np.clip(
                    round(task["candidate_frame"] + offset * task["video_fps"]),
                    0, task["video_frame_count"] - 1,
                ))
                video_time_sec = frame / task["video_fps"]
                status = "selected"
            selections[task["task_id"]] = frame
            decisions.append({
                "task_id": task["task_id"], "status": status,
                "selection_source": source,
                "candidate_frame": task["candidate_frame"],
                "offset_sec": offset, "video_time_sec": video_time_sec,
                "selected_frame": frame,
            })
        output = Path(args.selections_out).expanduser().resolve()
        _atomic_json(output, selections)
        _atomic_json(output.with_name(f"{output.stem}_decisions.json"), {
            "offset_config": str(Path(args.offset_config).expanduser().resolve()),
            "selected": sum(value is not None for value in selections.values()),
            "pending": sum(value is None for value in selections.values()),
            "decisions": decisions,
        })
        print(
            f"[wrote] {output} selected={sum(value is not None for value in selections.values())} "
            f"pending={sum(value is None for value in selections.values())}"
        )
        return 0
    if not args.selections:
        raise ValueError("export requires --selections")
    selections = _read_json(Path(args.selections).expanduser().resolve())
    if not isinstance(selections, dict):
        raise ValueError("selections must be a JSON object mapping task IDs to frame indices")
    by_id = {task["task_id"]: task for task in tasks}
    unknown = sorted(set(selections) - set(by_id))
    if unknown:
        raise KeyError(f"Unknown task IDs: {unknown}")
    pending = sorted(task_id for task_id, value in selections.items() if value is None)
    if pending:
        raise ValueError(f"Selections still pending manual review: {pending}")
    for task_id, frame_index in sorted(selections.items()):
        _export(
            root, by_id[task_id], int(frame_index), args.overwrite,
            args.allow_missing_arm_state,
        )
        print(f"[exported] {task_id} frame={int(frame_index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
