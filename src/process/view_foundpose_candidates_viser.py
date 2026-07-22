#!/usr/bin/env python3
"""Inspect FoundPose candidate-bank poses without running GoTrack.

The viewer loads the static poses already written to ``candidate_bank.json``
and places them in the calibrated capture scene.  Each rank has a distinct
color and an independent visibility checkbox, so symmetric pose hypotheses
can be compared before paying for a full GoTrack run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.process.view_gotrack_viser import (
    VideoFrames,
    _as_4x4,
    _evenly_spaced,
    _load_calibration,
    _load_robot_qpos,
    _video_shape,
    _wxyz,
)


COLORS = (
    (239, 83, 80),
    (52, 168, 83),
    (66, 133, 244),
    (244, 180, 0),
    (171, 71, 188),
)


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No candidates in {path}")
    candidates: list[dict[str, Any]] = []
    for rank, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        try:
            candidates.append({**row, "rank": rank, "pose_world": _as_4x4(row["pose_world"])})
        except (KeyError, TypeError, ValueError):
            continue
    if not candidates:
        raise ValueError(f"No valid pose_world candidates in {path}")
    return candidates


def _infer_frame_index(candidate_bank: Path) -> int:
    result_path = candidate_bank.with_name("result.json")
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            match = re.search(r"foundpose_frame_(\d+)", str(result.get("frame_dir", "")))
            if match:
                return int(match.group(1))
        except (OSError, ValueError, TypeError):
            pass
    return 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--frame-index", type=int, default=None,
                        help="Capture frame shown with the candidates; default: infer from result.json, else 30.")
    parser.add_argument("--video-dir", default=None, help="Default: <capture-dir>/undistorted_video")
    parser.add_argument("--camera-ids", nargs="*", default=None)
    parser.add_argument("--max-cameras", type=int, default=0,
                        help="Evenly sample this many cameras; 0 (default) shows all calibrated cameras.")
    parser.add_argument("--camera-image-max-side", type=int, default=480)
    parser.add_argument("--camera-scale", type=float, default=0.08)
    parser.add_argument("--show-camera-images", action="store_true",
                        help="Attach the selected capture frame to camera frustums; off by default.")
    parser.add_argument("--coordinate-frame", choices=("robot", "world"), default="robot")
    parser.add_argument("--c2r", default=None)
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--robot-urdf", default=str(Path.home() / "paradex/rsc/robot/xarm_inspire_DFTP.urdf"))
    parser.add_argument("--opacity", type=float, default=0.75)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cameras < 0 or args.camera_image_max_side < 32 or not 0.0 < args.opacity <= 1.0:
        raise ValueError("camera parameters and opacity must be positive")
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    mesh_path = Path(args.object_mesh).expanduser().resolve()
    bank_path = Path(args.candidate_bank).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else capture_dir / "undistorted_video"
    if not capture_dir.is_dir() or not mesh_path.is_file() or not bank_path.is_file():
        raise FileNotFoundError("--capture-dir, --object-mesh, and --candidate-bank must exist")

    candidates = _load_candidates(bank_path)
    frame_index = args.frame_index if args.frame_index is not None else _infer_frame_index(bank_path)
    intrinsics, extrinsics = _load_calibration(capture_dir)
    c2r_path = Path(args.c2r).expanduser().resolve() if args.c2r else capture_dir / "C2R.npy"
    if args.coordinate_frame == "robot":
        if not c2r_path.is_file():
            raise FileNotFoundError(f"Robot-coordinate view requires C2R.npy: {c2r_path}")
        world_from_robot = _as_4x4(np.load(c2r_path))
        view_from_world = np.linalg.inv(world_from_robot)
    else:
        world_from_robot = _as_4x4(np.load(c2r_path)) if c2r_path.is_file() else None
        view_from_world = np.eye(4)

    all_serials = [serial for serial in sorted(intrinsics) if serial in extrinsics]
    if args.camera_ids is not None:
        missing = [serial for serial in args.camera_ids if serial not in all_serials]
        if missing:
            raise ValueError(f"Unknown or uncalibrated camera IDs: {missing}")
        serials = list(args.camera_ids)
    else:
        serials = all_serials if args.max_cameras == 0 else _evenly_spaced(all_serials, args.max_cameras)
    if args.show_camera_images:
        serials = [serial for serial in serials if _video_shape(video_dir / f"{serial}.avi") is not None]
    if not serials:
        raise ValueError("No usable calibrated cameras")

    print(f"[candidate-viewer] frame={frame_index} candidates={len(candidates)} cameras={len(serials)}")
    for row in candidates:
        print(
            f"  rank={row['rank']} serial={row.get('source_serial')} "
            f"score={float(row.get('hybrid_score', float('nan'))):.4f} "
            f"iou={float(row.get('mean_iou', float('nan'))):.4f} "
            f"pnp={float(row.get('pnp_quality', float('nan'))):.1f} "
            f"consensus={float(row.get('consensus', float('nan'))):.4f} "
            f"dino={float(row.get('dino_reprojection_score', float('nan'))):.4f}"
        )
    if args.dry_run:
        return 0

    try:
        import trimesh
        import viser
    except ModuleNotFoundError as exc:
        if exc.name == "viser":
            raise RuntimeError("Viser is not installed. Run: conda run -n gotrack pip install viser") from exc
        raise

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a Trimesh from {mesh_path}")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", show_axes=True, axes_length=0.12, axes_radius=0.003)
    try:
        server.scene.add_grid("/world/grid", width=2.0, height=2.0, cell_size=0.1, section_size=0.5)
    except Exception:
        pass

    candidate_handles: list[tuple[Any, Any]] = []
    for row in candidates:
        rank = int(row["rank"])
        pose = view_from_world @ row["pose_world"]
        frame = server.scene.add_frame(
            f"/candidates/rank_{rank}",
            show_axes=True,
            axes_length=0.05,
            axes_radius=0.002,
            wxyz=_wxyz(pose[:3, :3]),
            position=tuple(float(value) for value in pose[:3, 3]),
        )
        mesh_handle = server.scene.add_mesh_simple(
            f"/candidates/rank_{rank}/mesh",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            color=COLORS[rank % len(COLORS)],
            opacity=args.opacity,
            side="double",
        )
        candidate_handles.append((frame, mesh_handle))

    readers = VideoFrames(video_dir, serials, args.camera_image_max_side) if args.show_camera_images else None
    for serial in serials:
        video_size = _video_shape(video_dir / f"{serial}.avi")
        if video_size is None:
            continue
        width, height = video_size
        view_from_camera = view_from_world @ np.linalg.inv(extrinsics[serial])
        fov = float(2.0 * np.arctan2(height / 2.0, intrinsics[serial][1, 1]))
        image = readers.read(serial, frame_index) if readers is not None else None
        handle = server.scene.add_camera_frustum(
            f"/cameras/{serial}",
            fov=fov,
            aspect=float(width / height),
            scale=args.camera_scale,
            wxyz=_wxyz(view_from_camera[:3, :3]),
            position=tuple(view_from_camera[:3, 3]),
            color=(255, 190, 80),
            image=image,
        )

        @handle.on_click
        def _(_, wxyz=_wxyz(view_from_camera[:3, :3]), position=tuple(view_from_camera[:3, 3])):
            for client in server.get_clients().values():
                client.camera.wxyz = wxyz
                client.camera.position = position

    robot_vis = None
    if not args.no_robot:
        urdf_path = Path(args.robot_urdf).expanduser().resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Robot URDF is missing: {urdf_path}")
        try:
            paradex_root = Path.home() / "paradex"
            if str(paradex_root) not in sys.path:
                sys.path.insert(0, str(paradex_root))
            from viser.extras import ViserUrdf
            qpos = _load_robot_qpos(capture_dir, frame_index + 1)
            robot_base = server.scene.add_frame("/robot", show_axes=True, axes_length=0.12, axes_radius=0.003)
            if args.coordinate_frame == "world":
                if world_from_robot is None:
                    raise FileNotFoundError("World-coordinate robot view requires C2R.npy")
                robot_base.position = tuple(world_from_robot[:3, 3])
                robot_base.wxyz = _wxyz(world_from_robot[:3, :3])
            robot_vis = ViserUrdf(server, urdf_path, root_node_name="/robot")
            robot_vis.update_cfg(qpos[frame_index])
        except Exception as exc:
            raise RuntimeError("Could not load the recorded robot; use --no-robot to inspect candidates only") from exc

    with server.gui.add_folder("FoundPose candidates"):
        visibility_controls = []
        for row in candidates:
            rank = int(row["rank"])
            display_score = float(row.get("asymmetry_combined_score", row.get("hybrid_score", float("nan"))))
            label = (
                f"rank {rank}  score={display_score:.3f}  "
                f"IoU={float(row.get('mean_iou', float('nan'))):.3f}  "
                f"DINO={float(row.get('dino_reprojection_score', float('nan'))):.3f}"
            )
            visibility_controls.append(server.gui.add_checkbox(label, initial_value=True))
        show_robot = server.gui.add_checkbox("Show robot", initial_value=robot_vis is not None)
        server.gui.add_text("Frame", initial_value=str(frame_index), disabled=True)

    def update_visibility() -> None:
        for control, (frame, mesh_handle) in zip(visibility_controls, candidate_handles):
            frame.visible = bool(control.value)
            mesh_handle.visible = bool(control.value)
        if robot_vis is not None:
            robot_vis.show_visual = bool(show_robot.value)

    for control in visibility_controls:
        @control.on_update
        def _(_event: Any) -> None:
            update_visibility()
    @show_robot.on_update
    def _(_event: Any) -> None:
        update_visibility()

    update_visibility()
    print(f"[candidate-viewer] open http://localhost:{args.port} (or this host's IP:{args.port}); Ctrl+C to stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        if readers is not None:
            readers.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
