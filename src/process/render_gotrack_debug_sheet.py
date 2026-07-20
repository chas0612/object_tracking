#!/usr/bin/env python3
"""Create one sparse reprojection contact sheet for fast GoTrack QA.

Unlike a grid video, this reads only a few selected RGB frames and writes one
JPEG/PNG.  By default it renders six evenly-spaced cameras at the first,
middle, and last tracked frame: 18 mesh overlays total and no intermediate
files.  It is intended as the first visual check before opening the Viser
viewer or generating a short video for a suspicious interval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.overlay_object_video_single import ObjectOverlayRenderer, load_cam_param


def _load_poses(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses: dict[int, np.ndarray] = {}
    for row in data:
        if not isinstance(row, dict) or row.get("pose_world") is None:
            continue
        pose = np.asarray(row["pose_world"], dtype=np.float64)
        if pose.shape == (3, 4):
            pose = np.vstack([pose, [0, 0, 0, 1]])
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            poses[int(row["frame_index"])] = pose
    if not poses:
        raise ValueError(f"No finite pose_world records in {path}")
    return poses


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in dict.fromkeys(indices)]


def _read_frame(video_path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {index} from {video_path}")
        return image
    finally:
        cap.release()


def _pose_at(poses: dict[int, np.ndarray], frame: int) -> np.ndarray:
    earlier = [index for index in poses if index <= frame]
    return poses[max(earlier)] if earlier else poses[min(poses)]


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 250), 28), (0, 0, 0), -1)
    cv2.putText(out, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--gotrack-records", required=True)
    parser.add_argument("--output", required=True, help="New .jpg/.jpeg/.png file; never overwritten.")
    parser.add_argument("--video-dir", default=None, help="Default: <capture-dir>/undistorted_video")
    parser.add_argument("--camera-ids", nargs="*", default=None)
    parser.add_argument("--max-cameras", type=int, default=6)
    parser.add_argument("--frame-indices", nargs="*", type=int, default=None,
                        help="Default: first, midpoint, last tracked frame.")
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cameras < 1 or args.cell_width < 64 or not 0.0 < args.alpha <= 1.0:
        raise ValueError("invalid camera count, cell width, or alpha")
    capture = Path(args.capture_dir).expanduser().resolve()
    mesh_path = Path(args.object_mesh).expanduser().resolve()
    records_path = Path(args.gotrack_records).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else capture / "undistorted_video"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not capture.is_dir() or not mesh_path.is_file() or not records_path.is_file() or not video_dir.is_dir():
        raise FileNotFoundError("capture, mesh, records, and video directory must exist")

    poses = _load_poses(records_path)
    intrinsics, extrinsics = load_cam_param(capture / "cam_param")
    available = sorted(path.stem for path in video_dir.glob("*.avi"))
    candidates = [serial for serial in available if serial in intrinsics and serial in extrinsics]
    serials = list(args.camera_ids) if args.camera_ids is not None else _evenly_spaced(candidates, args.max_cameras)
    invalid = [serial for serial in serials if serial not in candidates]
    if invalid:
        raise ValueError(f"Unusable camera IDs: {invalid}")
    if not serials:
        raise RuntimeError("No usable camera videos")
    first, last = min(poses), max(poses)
    frames = args.frame_indices or [first, (first + last) // 2, last]
    frames = list(dict.fromkeys(max(first, min(last, int(frame))) for frame in frames))
    print(f"[sheet] cameras={serials}; frames={frames}; output={output}")
    if args.dry_run:
        return 0

    probe = _read_frame(video_dir / f"{serials[0]}.avi", frames[0])
    height, width = probe.shape[:2]
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh: {mesh_path}")
    selected_intrinsics = {serial: intrinsics[serial] for serial in serials}
    selected_extrinsics = {serial: extrinsics[serial] for serial in serials}
    renderer = ObjectOverlayRenderer(mesh, selected_intrinsics, selected_extrinsics, height, width, alpha=args.alpha)
    cell_height = round(height * args.cell_width / width)
    rows: list[np.ndarray] = []
    for frame in frames:
        images = [_read_frame(video_dir / f"{serial}.avi", frame) for serial in renderer.serials]
        overlays = renderer.render(_pose_at(poses, frame), images)
        cells = [_label(cv2.resize(image, (args.cell_width, cell_height), interpolation=cv2.INTER_AREA),
                        f"{serial}  f{frame}") for serial, image in zip(renderer.serials, overlays)]
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality] if output.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(output), sheet, params):
        raise RuntimeError(f"Could not write {output}")
    print(f"[sheet] wrote {output} ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
