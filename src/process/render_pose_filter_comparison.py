#!/usr/bin/env python3
"""Render camera evidence and filtered GoTrack trajectories side by side.

The left panel is the unmodified camera frame and is visual evidence, not 6D
ground truth. An optional raw-GoTrack panel and the two filtered panels render
the same object mesh, making temporal lag, over-smoothing, and silhouette
misalignment directly visible against both the source track and camera evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
    rows = json.loads(path.read_text(encoding="utf-8"))
    poses: dict[int, np.ndarray] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("pose_world") is None:
            continue
        pose = np.asarray(row["pose_world"], dtype=np.float64)
        if pose.shape == (3, 4):
            pose = np.vstack([pose, [0.0, 0.0, 0.0, 1.0]])
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            poses[int(row["frame_index"])] = pose
    if not poses:
        raise ValueError(f"No finite pose_world entries in {path}")
    return poses


def _pose_at(poses: dict[int, np.ndarray], frame: int) -> np.ndarray:
    if frame in poses:
        return poses[frame]
    earlier = [index for index in poses if index <= frame]
    return poses[max(earlier)] if earlier else poses[min(poses)]


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {path}")
    return mesh


def _panel(image: np.ndarray, label: str, width: int) -> np.ndarray:
    height = round(image.shape[0] * width / image.shape[1])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    bar_height = max(32, round(height * 0.055))
    cv2.rectangle(resized, (0, 0), (width, bar_height), (0, 0, 0), -1)
    font_scale = max(0.55, width / 1000.0)
    cv2.putText(
        resized, label, (10, round(bar_height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (255, 255, 255), max(1, round(width / 600)), cv2.LINE_AA,
    )
    return resized


def _crop(image: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return image
    x, y, width, height = crop
    return image[y:y + height, x:x + width]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--raw-records", default=None,
                        help="Optional original GoTrack records, shown after camera evidence.")
    parser.add_argument("--gaussian-records", required=True)
    parser.add_argument("--one-euro-records", required=True)
    parser.add_argument("--gaussian-label", default="GAUSSIAN")
    parser.add_argument("--one-euro-label", default="ONE EURO")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output", required=True, help="Output MP4; existing files are not replaced.")
    parser.add_argument("--video-dir", default=None,
                        help="Default: <capture-dir>/undistorted_video")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Inclusive; default is video end.")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--cell-width", type=int, default=640)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--output-fps", type=float, default=None,
                        help="Default: source video FPS.")
    parser.add_argument("--crf", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_frame < 0 or args.cell_width < 64 or not 0.0 < args.alpha <= 1.0:
        raise ValueError("Invalid frame, width, or alpha setting")
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    video_dir = (Path(args.video_dir).expanduser().resolve() if args.video_dir
                 else capture_dir / "undistorted_video")
    mesh_path = Path(args.object_mesh).expanduser().resolve()
    raw_path = Path(args.raw_records).expanduser().resolve() if args.raw_records else None
    gaussian_path = Path(args.gaussian_records).expanduser().resolve()
    one_euro_path = Path(args.one_euro_records).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    video_path = video_dir / f"{args.camera_id}.avi"
    required_paths = [capture_dir, video_dir, mesh_path, gaussian_path, one_euro_path, video_path]
    if raw_path is not None:
        required_paths.append(raw_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")

    raw = _load_poses(raw_path) if raw_path is not None else None
    gaussian = _load_poses(gaussian_path)
    one_euro = _load_poses(one_euro_path)
    intrinsics, extrinsics = load_cam_param(capture_dir / "cam_param")
    if args.camera_id not in intrinsics or args.camera_id not in extrinsics:
        raise KeyError(f"Camera {args.camera_id} is missing calibration")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    last_frame = source_frames - 1 if args.end_frame is None else args.end_frame
    pose_last_frames = [max(gaussian), max(one_euro)]
    if raw is not None:
        pose_last_frames.append(max(raw))
    last_frame = min(last_frame, source_frames - 1, *pose_last_frames)
    if last_frame < args.start_frame:
        raise ValueError("Selected frame range is empty")

    crop = tuple(args.crop) if args.crop else None
    if crop is not None:
        x, y, width, height = crop
        if min(x, y) < 0 or min(width, height) < 1 or x + width > source_width or y + height > source_height:
            raise ValueError(f"Crop {crop} is outside {source_width}x{source_height}")
    cropped_width = crop[2] if crop else source_width
    cropped_height = crop[3] if crop else source_height
    cell_height = round(cropped_height * args.cell_width / cropped_width)
    output_width = args.cell_width * (4 if raw is not None else 3)
    output_height = cell_height
    if output_height % 2:
        output_height += 1
    output_fps = args.output_fps or source_fps

    mesh = _load_mesh(mesh_path)
    selected_intrinsics = {args.camera_id: intrinsics[args.camera_id]}
    selected_extrinsics = {args.camera_id: extrinsics[args.camera_id]}
    raw_renderer = None
    if raw is not None:
        raw_renderer = ObjectOverlayRenderer(
            mesh, selected_intrinsics, selected_extrinsics, source_height, source_width,
            color_bgr=(255, 180, 0), alpha=args.alpha,
        )
    gaussian_renderer = ObjectOverlayRenderer(
        mesh, selected_intrinsics, selected_extrinsics, source_height, source_width,
        color_bgr=(0, 165, 255), alpha=args.alpha,
    )
    one_euro_renderer = ObjectOverlayRenderer(
        mesh, selected_intrinsics, selected_extrinsics, source_height, source_width,
        color_bgr=(0, 255, 0), alpha=args.alpha,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{output_width}x{output_height}",
        "-r", str(output_fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", str(args.crf), "-pix_fmt", "yuv420p", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Could not create ffmpeg input pipe")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    total = last_frame - args.start_frame + 1
    try:
        for ordinal, frame_index in enumerate(range(args.start_frame, last_frame + 1), start=1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
            raw_overlay = (
                raw_renderer.render(_pose_at(raw, frame_index), [frame])[0]
                if raw is not None and raw_renderer is not None else None
            )
            gaussian_overlay = gaussian_renderer.render(_pose_at(gaussian, frame_index), [frame])[0]
            one_euro_overlay = one_euro_renderer.render(_pose_at(one_euro, frame_index), [frame])[0]
            panels = [_panel(
                _crop(frame, crop), f"CAMERA EVIDENCE  f{frame_index}", args.cell_width
            )]
            if raw_overlay is not None:
                panels.append(_panel(_crop(raw_overlay, crop), "RAW GOTRACK", args.cell_width))
            panels.extend([
                _panel(_crop(gaussian_overlay, crop), args.gaussian_label, args.cell_width),
                _panel(_crop(one_euro_overlay, crop), args.one_euro_label, args.cell_width),
            ])
            comparison = np.concatenate(panels, axis=1)
            if comparison.shape[0] != output_height:
                comparison = cv2.copyMakeBorder(
                    comparison, 0, output_height - comparison.shape[0], 0, 0,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0),
                )
            process.stdin.write(comparison.tobytes())
            if ordinal == 1 or ordinal % 30 == 0 or ordinal == total:
                print(f"[compare] {ordinal}/{total} frame={frame_index}", flush=True)
        process.stdin.close()
        return_code = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with {return_code}: {stderr.strip()}")
    except Exception:
        process.kill()
        output.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
    print(f"[compare] wrote {output} ({output_width}x{output_height}@{output_fps:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
