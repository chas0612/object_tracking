#!/usr/bin/env python3
"""Export canonical GoTrack trajectories into final-dataset NPZ files.

The source of truth is a scheduler task directory and/or ``latest_trials.json``.
Each selected ``world_pose_records.json`` is converted to the dataset
convention: one float32 4x4 world pose per ``frame_<index>`` key. The expected
frame count comes from an existing ``object_6d_pose.npz``, a human capture's
contiguous ``object_6d/pose_*.txt`` sequence, or the published videos themselves.
No contract source is modified.

Use ``--frame-contract video`` when the poses must line up with the published
videos rather than with an older pose file. The ``allegro_v5`` captures need it:
their ``vid/*.mp4`` re-encode dropped the first three frames of the original AVI,
but the existing ``object_6d_pose.npz`` was tracked before that trim and still
carries the untrimmed length, so it is three frames out of step with the videos
the new run tracks.

The default is a read-only audit. Pass ``--write`` to atomically create the
new file in each eligible episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TARGET = Path.home() / "shared_data/capture/eccv2026/v0/inspire_dftp"
DEFAULT_MANIFEST = (
    Path.home()
    / "shared_data/object_tracking/gotrack_debug_sheets/inspire_dftp_latest/latest_trials.json"
)
DEFAULT_SHARED_ROOT = Path.home() / "shared_data"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_holds(path: Path | None) -> tuple[set[str], set[tuple[str, str]]]:
    if path is None:
        return set(), set()
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Hold config must be a JSON object: {path}")
    all_episodes = payload.get("all_episodes", [])
    episodes = payload.get("episodes", {})
    if not isinstance(all_episodes, list) or not all(isinstance(x, str) for x in all_episodes):
        raise ValueError("'all_episodes' must be a list of object names")
    if not isinstance(episodes, dict):
        raise ValueError("'episodes' must map object names to episode lists")
    held_pairs: set[tuple[str, str]] = set()
    for object_name, values in episodes.items():
        if not isinstance(object_name, str) or not isinstance(values, list):
            raise ValueError("'episodes' must map object names to episode lists")
        held_pairs.update((object_name, str(value)) for value in values)
    return set(all_episodes), held_pairs


def _load_selection(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Selection config must be a JSON object: {path}")
    selected: set[tuple[str, str]] = set()
    for object_name, episodes in payload.items():
        if not isinstance(object_name, str) or not isinstance(episodes, list):
            raise ValueError("Selection config must map object names to episode lists")
        selected.update((object_name, str(episode)) for episode in episodes)
    return selected


def _schedule_entries(schedule_dir: Path, shared_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_path = schedule_dir / "manifest.json"
    tasks_dir = schedule_dir / "tasks"
    if not manifest_path.is_file() or not tasks_dir.is_dir():
        raise FileNotFoundError(f"Schedule has no manifest/tasks: {schedule_dir}")
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for task_path in sorted(tasks_dir.glob("*.json")):
        task = _read_json(task_path)
        object_name, episode = str(task.get("object_name")), str(task.get("episode"))
        attempt_dir = task.get("attempt_dir")
        if not isinstance(attempt_dir, str):
            raise ValueError(f"Task has no attempt_dir: {task_path}")
        record = (
            shared_root
            / attempt_dir
            / "gotrack_tracking"
            / "gotrack_output"
            / object_name
            / "world_pose_records.json"
        )
        key = (object_name, episode)
        if key in entries:
            raise ValueError(f"Duplicate schedule task: {object_name}/{episode}")
        entries[key] = {
            "object_name": object_name,
            "episode": episode,
            "latest_schedule": schedule_dir.name,
            "latest_status": task.get("status"),
            "attempt_dir": attempt_dir,
            "record": str(record),
        }
    return entries


def _existing_frame_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as archive:
        indices: list[int] = []
        for key in archive.files:
            if not key.startswith("frame_"):
                raise ValueError(f"Unexpected key {key!r} in {path}")
            try:
                indices.append(int(key.removeprefix("frame_")))
            except ValueError as exc:
                raise ValueError(f"Invalid frame key {key!r} in {path}") from exc
        indices.sort()
        if indices != list(range(len(indices))):
            raise ValueError(f"Existing NPZ keys are not contiguous in {path}")
        for index in (indices[:1] + indices[-1:]):
            pose = np.asarray(archive[f"frame_{index}"])
            if pose.shape != (4, 4):
                raise ValueError(f"Existing frame_{index} has shape {pose.shape} in {path}")
        return len(indices)


def _text_pose_frame_count(path: Path) -> int:
    if not path.is_dir():
        raise FileNotFoundError(path)
    indices: list[int] = []
    for pose_path in path.glob("pose_*.txt"):
        suffix = pose_path.stem.removeprefix("pose_")
        try:
            indices.append(int(suffix))
        except ValueError as exc:
            raise ValueError(f"Invalid text pose file: {pose_path}") from exc
    indices.sort()
    if not indices:
        raise ValueError(f"No pose_*.txt files under {path}")
    if indices != list(range(len(indices))):
        raise ValueError(f"Text pose indices are not contiguous under {path}")
    for index in (indices[0], indices[-1]):
        pose = np.loadtxt(path / f"pose_{index:06d}.txt")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(
                f"Text pose_{index:06d} is not a finite 4x4 matrix under {path}"
            )
        if not np.allclose(pose[3], np.array([0, 0, 0, 1]), atol=1e-5):
            raise ValueError(f"Text pose_{index:06d} has an invalid last row")
    return len(indices)


def _video_frame_count(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value or value == "N/A":
        raise ValueError(f"ffprobe could not count frames in {path}: {value!r}")
    return int(value)


def _video_frame_contract(video_dir: Path, probe_cameras: int) -> int:
    """Frame count shared by the published videos the tracker actually reads."""
    if not video_dir.is_dir():
        raise FileNotFoundError(video_dir)
    videos = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi"))
    if not videos:
        raise ValueError(f"No videos under {video_dir}")
    probed = videos if probe_cameras <= 0 else videos[:probe_cameras]
    counts = {path.name: _video_frame_count(path) for path in probed}
    distinct = set(counts.values())
    if len(distinct) != 1:
        raise ValueError(f"Videos disagree on frame count: {counts}")
    return distinct.pop()


def _frame_contract(
    episode_dir: Path,
    mode: str,
    original_name: str,
    text_pose_dir: str,
    video_dir: str,
    video_probe_cameras: int,
) -> tuple[int, str, Path]:
    npz_path = episode_dir / original_name
    text_path = episode_dir / text_pose_dir
    video_path = episode_dir / video_dir
    # ``video`` is never part of ``auto``: an episode can hold both an older pose
    # file and the videos it disagrees with, so choosing between them must be
    # explicit.
    if mode == "video":
        return _video_frame_contract(video_path, video_probe_cameras), "video", video_path
    if mode in {"auto", "npz"} and npz_path.is_file():
        return _existing_frame_count(npz_path), "npz", npz_path
    if mode in {"auto", "text"} and text_path.is_dir():
        return _text_pose_frame_count(text_path), "text", text_path
    raise FileNotFoundError(
        f"No {mode} frame contract in {episode_dir}: "
        f"npz={npz_path}, text={text_path}"
    )


def _load_records(
    path: Path, expected_frames: int, max_boundary_fill: int
) -> tuple[dict[str, np.ndarray], list[int]]:
    records = _read_json(path)
    if not isinstance(records, list):
        raise ValueError("record root is not a list")
    indexed: dict[int, np.ndarray] = {}
    seen_indices: set[int] = set()
    for record_position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {record_position}: record is not an object")
        frame_index = record.get("frame_index")
        if not isinstance(frame_index, int) or not 0 <= frame_index < expected_frames:
            raise ValueError(f"record {record_position}: invalid frame_index={frame_index!r}")
        if frame_index in seen_indices:
            raise ValueError(f"frame {frame_index}: duplicate record")
        seen_indices.add(frame_index)
        if record.get("pose_world") is None:
            continue
        pose = np.asarray(record["pose_world"], dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"frame {frame_index}: pose is not a finite 4x4 matrix")
        if not np.allclose(pose[3], np.array([0, 0, 0, 1], dtype=np.float32), atol=1e-5):
            raise ValueError(f"frame {frame_index}: invalid homogeneous last row")
        indexed[frame_index] = pose
    if not indexed:
        raise ValueError("no valid poses")

    missing = sorted(set(range(expected_frames)) - set(indexed))
    first_valid, last_valid = min(indexed), max(indexed)
    internal = [index for index in missing if first_valid < index < last_valid]
    if internal:
        raise ValueError(f"internal pose gaps: {internal[:10]}")
    if len(missing) > max_boundary_fill:
        raise ValueError(
            f"boundary pose gaps {len(missing)} exceed --max-boundary-fill={max_boundary_fill}"
        )
    for index in range(first_valid):
        indexed[index] = indexed[first_valid].copy()
    for index in range(last_valid + 1, expected_frames):
        indexed[index] = indexed[last_valid].copy()
    poses = {f"frame_{index}": indexed[index] for index in range(expected_frames)}
    return poses, missing


def _atomic_savez_compressed(path: Path, poses: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **poses)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--latest-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--no-latest-manifest",
        action="store_true",
        help="Use only --schedule-dir entries. Recommended for a new campaign.",
    )
    parser.add_argument(
        "--schedule-dir",
        action="append",
        default=[],
        help="Completed scheduler directory whose tasks override matching latest-manifest entries.",
    )
    parser.add_argument(
        "--object-episodes-json",
        default=None,
        help="Optional exact object-to-episode selection; other target episodes are ignored.",
    )
    parser.add_argument("--holds-json", default=None)
    parser.add_argument("--original-name", default="object_6d_pose.npz")
    parser.add_argument("--text-pose-dir", default="object_6d")
    parser.add_argument(
        "--frame-contract",
        choices=("auto", "npz", "text", "video"),
        default="auto",
        help="Expected-frame source. auto prefers the original NPZ, then object_6d text "
             "files. video counts the published videos instead, for captures whose "
             "existing pose file predates a video re-encode.",
    )
    parser.add_argument(
        "--video-dir",
        default="vid",
        help="Episode-relative video directory used by --frame-contract video.",
    )
    parser.add_argument(
        "--video-probe-cameras",
        type=int,
        default=1,
        help="Videos to probe per episode for --frame-contract video. 0 probes all.",
    )
    parser.add_argument("--output-name", default="object_6d_pose_v2.npz")
    parser.add_argument("--report", default=None, help="Optional JSON audit/provenance report.")
    parser.add_argument(
        "--max-boundary-fill",
        type=int,
        default=0,
        help="Fill up to this many missing leading/trailing poses from the nearest valid pose.",
    )
    parser.add_argument("--write", action="store_true", help="Create outputs; default is dry-run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output atomically.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_root = Path(args.target_root).expanduser().resolve()
    shared_root = Path(args.shared_root).expanduser().resolve()
    manifest_path = (
        None
        if args.no_latest_manifest
        else Path(args.latest_manifest).expanduser().resolve()
    )
    holds_path = Path(args.holds_json).expanduser().resolve() if args.holds_json else None
    selection_path = (
        Path(args.object_episodes_json).expanduser().resolve()
        if args.object_episodes_json
        else None
    )
    report_path = Path(args.report).expanduser().resolve() if args.report else None
    if not target_root.is_dir() or not shared_root.is_dir():
        raise FileNotFoundError("--target-root and --shared-root must exist")
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"--latest-manifest does not exist: {manifest_path}")
    if holds_path is not None and not holds_path.is_file():
        raise FileNotFoundError(f"Hold config does not exist: {holds_path}")
    if selection_path is not None and not selection_path.is_file():
        raise FileNotFoundError(f"Selection config does not exist: {selection_path}")
    if (
        Path(args.original_name).name != args.original_name
        or Path(args.output_name).name != args.output_name
        or Path(args.text_pose_dir).name != args.text_pose_dir
    ):
        raise ValueError("Contract/output names must be plain file names")
    if args.max_boundary_fill < 0:
        raise ValueError("--max-boundary-fill must be non-negative")

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if manifest_path is not None:
        manifest = _read_json(manifest_path)
        entries = manifest.get("episodes")
        if not isinstance(entries, list):
            raise ValueError(f"Manifest has no episode list: {manifest_path}")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = (str(entry.get("object_name")), str(entry.get("episode")))
            if key in latest:
                raise ValueError(f"Duplicate latest-manifest entry: {key[0]}/{key[1]}")
            latest[key] = entry
    schedule_dirs = [Path(value).expanduser().resolve() for value in args.schedule_dir]
    if manifest_path is None and not schedule_dirs:
        raise ValueError("--no-latest-manifest requires at least one --schedule-dir")
    for schedule_dir in schedule_dirs:
        latest.update(_schedule_entries(schedule_dir, shared_root))
    held_objects, held_pairs = _load_holds(holds_path)
    selection = _load_selection(selection_path)

    results: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for episode_dir in sorted(path for path in target_root.glob("*/*") if path.is_dir()):
        object_name, episode = episode_dir.parent.name, episode_dir.name
        key = (object_name, episode)
        if selection is not None and key not in selection:
            continue
        output = episode_dir / args.output_name
        row: dict[str, Any] = {
            "object_name": object_name,
            "episode": episode,
            "output": str(output),
        }
        try:
            if object_name in held_objects or key in held_pairs:
                row.update(status="held")
            elif key not in latest:
                row.update(status="error", reason="latest trial entry missing")
            elif latest[key].get("latest_status") != "completed":
                row.update(
                    status="error",
                    reason=f"latest status is {latest[key].get('latest_status')!r}",
                )
            else:
                record_value = latest[key].get("record")
                record_path = Path(record_value).expanduser() if isinstance(record_value, str) else None
                if record_path is None or not record_path.is_file():
                    raise FileNotFoundError(f"latest record missing: {record_value!r}")
                expected_frames, contract_type, contract_path = _frame_contract(
                    episode_dir,
                    args.frame_contract,
                    args.original_name,
                    args.text_pose_dir,
                    args.video_dir,
                    args.video_probe_cameras,
                )
                poses, filled_frames = _load_records(
                    record_path, expected_frames, args.max_boundary_fill
                )
                if output.exists() and not args.overwrite:
                    row.update(status="exists", reason="output already exists")
                elif args.write:
                    _atomic_savez_compressed(output, poses)
                    row.update(
                        status="written",
                        frames=expected_frames,
                        frame_contract_type=contract_type,
                        frame_contract_path=str(contract_path),
                        boundary_filled_frames=filled_frames,
                        source_record=str(record_path),
                        source_schedule=latest[key].get("latest_schedule"),
                        sha256=_sha256(output),
                    )
                else:
                    row.update(
                        status="ready",
                        frames=expected_frames,
                        frame_contract_type=contract_type,
                        frame_contract_path=str(contract_path),
                        boundary_filled_frames=filled_frames,
                        source_record=str(record_path),
                        source_schedule=latest[key].get("latest_schedule"),
                    )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        results.append(row)
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "write" if args.write else "dry-run",
        "target_root": str(target_root),
        "shared_root": str(shared_root),
        "latest_manifest": str(manifest_path) if manifest_path else None,
        "schedule_dirs": [str(path) for path in schedule_dirs],
        "object_episodes_json": str(selection_path) if selection_path else None,
        "holds_json": str(holds_path) if holds_path else None,
        "original_name": args.original_name,
        "text_pose_dir": args.text_pose_dir,
        "frame_contract": args.frame_contract,
        "video_dir": args.video_dir,
        "output_name": args.output_name,
        "max_boundary_fill": args.max_boundary_fill,
        "counts": counts,
        "episodes": results,
    }
    print(json.dumps({"mode": report["mode"], "counts": counts}, indent=2))
    for row in results:
        if row["status"] in {"error", "exists"}:
            print(f"[{row['status']}] {row['object_name']}/{row['episode']}: {row.get('reason')}")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f".{report_path.name}.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
    return 1 if counts.get("error", 0) or counts.get("exists", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
