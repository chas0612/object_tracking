#!/usr/bin/env python3
"""Interactively inspect one GoTrack trajectory in Viser without rendering video.

The viewer keeps one object mesh in the calibrated world frame and updates its
pose from ``world_pose_records.json`` with a frame slider.  It also shows a
small, evenly-spaced subset of calibrated camera frustums.  Camera images are
off by default for responsive playback; they can be enabled on demand without
creating an overlay video, image sequence, or any new NAS output.

Example (from the repository root, in the gotrack environment)::

  conda run -n gotrack python -u src/process/view_gotrack_viser.py \\
    --capture-dir ~/shared_data/capture/eccv2026/inspire_dftp/attached_container/0 \\
    --object-mesh ~/shared_data/mesh_blender/attached_container/attached_container.obj \\
    --gotrack-records ~/shared_data/capture/eccv2026/inspire_dftp/attached_container/0/object_tracking_foundpose_gotrack/inspire_dftp_gotrack_01/attempt_01/gotrack_tracking/gotrack_output/attached_container/world_pose_records.json

Open the printed URL.  Clicking a camera frustum moves the browser camera to
that view; use the timeline slider or Play to inspect the object motion.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def _as_4x4(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected a finite 3x4 or 4x4 transform, got {matrix.shape}")
    return matrix


def _wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a rotation matrix without requiring scipy."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
            qw = (rotation[2, 1] - rotation[1, 2]) / scale; qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale; qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
            qw = (rotation[0, 2] - rotation[2, 0]) / scale; qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale; qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
            qw = (rotation[1, 0] - rotation[0, 1]) / scale; qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale; qz = 0.25 * scale
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(x) for x in quat)


def _load_records(path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    poses: dict[int, np.ndarray] = {}
    for row in payload:
        if not isinstance(row, dict) or row.get("pose_world") is None:
            continue
        try:
            poses[int(row["frame_index"])] = _as_4x4(row["pose_world"])
        except (KeyError, TypeError, ValueError):
            continue
    if not poses:
        raise ValueError(f"No valid pose_world entries in {path}")
    return poses


def _load_calibration(capture_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    intr_raw = json.loads((capture_dir / "cam_param" / "intrinsics.json").read_text(encoding="utf-8"))
    extr_raw = json.loads((capture_dir / "cam_param" / "extrinsics.json").read_text(encoding="utf-8"))
    intrinsics: dict[str, np.ndarray] = {}
    extrinsics: dict[str, np.ndarray] = {}
    for serial in sorted(set(intr_raw) & set(extr_raw)):
        try:
            intrinsics[serial] = np.asarray(intr_raw[serial]["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
            extrinsics[serial] = _as_4x4(extr_raw[serial])  # world -> camera
        except (KeyError, TypeError, ValueError):
            continue
    return intrinsics, extrinsics


def _evenly_spaced(items: list[str], maximum: int) -> list[str]:
    if len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, maximum).round().astype(int)
    return [items[index] for index in dict.fromkeys(indices)]


def _video_shape(video_path: Path) -> tuple[int, int] | None:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height) if width > 0 and height > 0 else None
    finally:
        cap.release()


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _load_robot_qpos(capture_dir: Path, frame_count: int) -> np.ndarray:
    """Interpolate Inspire arm/hand recordings at camera-frame timestamps."""
    raw = capture_dir / "raw"
    arm_time = np.asarray(np.load(raw / "arm" / "time.npy", allow_pickle=True), dtype=np.float64)
    arm_qpos = np.asarray(np.load(raw / "arm" / "position.npy", allow_pickle=True), dtype=np.float64)
    hand_time = np.asarray(np.load(raw / "hand" / "time.npy", allow_pickle=True), dtype=np.float64)
    hand_raw = np.asarray(np.load(raw / "hand" / "position.npy", allow_pickle=True), dtype=np.float64)
    frame_times = np.asarray(np.load(raw / "timestamps" / "timestamp.npy", allow_pickle=True), dtype=np.float64)
    if arm_qpos.ndim != 2 or arm_qpos.shape[1] != 6 or hand_raw.ndim != 2 or hand_raw.shape[1] != 6:
        raise ValueError(f"Expected 6-DoF arm and hand streams, got arm={arm_qpos.shape}, hand={hand_raw.shape}")
    if len(arm_time) < 2 or len(hand_time) < 2 or len(frame_times) < 2:
        raise ValueError("Robot and camera timestamp streams must have at least two entries")
    # Controller order: little, ring, middle, index, thumb_2, thumb_1.
    hand = np.empty_like(hand_raw, dtype=np.float64)
    hand[:, 0] = 1.15 * (1.0 - hand_raw[:, 5] / 1000.0)
    hand[:, 1] = 0.55 * (1.0 - hand_raw[:, 4] / 1000.0)
    hand[:, 2] = 1.60 * (1.0 - hand_raw[:, 3] / 1000.0)
    hand[:, 3] = 1.60 * (1.0 - hand_raw[:, 2] / 1000.0)
    hand[:, 4] = 1.60 * (1.0 - hand_raw[:, 1] / 1000.0)
    hand[:, 5] = 1.60 * (1.0 - hand_raw[:, 0] / 1000.0)
    hand_at_arm = np.column_stack([np.interp(arm_time, hand_time, hand[:, i]) for i in range(6)])
    qpos_at_arm = np.concatenate([arm_qpos, hand_at_arm], axis=1)
    if len(frame_times) < frame_count:
        dt = float(np.median(np.diff(frame_times)))
        frame_times = np.concatenate([frame_times, frame_times[-1] + dt * np.arange(1, frame_count - len(frame_times) + 1)])
    return np.column_stack([np.interp(frame_times[:frame_count], arm_time, qpos_at_arm[:, i]) for i in range(12)])


class VideoFrames:
    """Lazy, bounded camera-frame reader; nothing is persisted to disk."""

    def __init__(self, video_dir: Path, serials: list[str], max_side: int):
        self.video_dir = video_dir
        self.max_side = max_side
        self.caps: dict[str, Any] = {}
        self.last_frame: dict[str, int] = {}

    def read(self, serial: str, frame_index: int) -> np.ndarray | None:
        import cv2
        cap = self.caps.get(serial)
        if cap is None:
            cap = cv2.VideoCapture(str(self.video_dir / f"{serial}.avi"))
            if not cap.isOpened():
                cap.release()
                return None
            self.caps[serial] = cap
        if self.last_frame.get(serial) != frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
        if not ok:
            return None
        self.last_frame[serial] = frame_index + 1
        height, width = bgr.shape[:2]
        scale = min(1.0, self.max_side / max(width, height))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        for cap in self.caps.values():
            cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--gotrack-records", required=True)
    parser.add_argument("--video-dir", default=None, help="Default: <capture-dir>/undistorted_video")
    parser.add_argument("--camera-ids", nargs="*", default=None, help="Optional calibrated camera serials to show.")
    parser.add_argument("--max-cameras", type=int, default=0,
                        help="Evenly sample this many cameras when --camera-ids is omitted; 0 (default) shows all cameras.")
    parser.add_argument("--frame-step", type=int, default=1, help="Timeline slider step in tracked frames.")
    parser.add_argument("--camera-image-max-side", type=int, default=480)
    parser.add_argument("--camera-image-stride", type=int, default=10,
                        help="Refresh camera-frustum images every N playback frames (object/robot still update every frame).")
    parser.add_argument("--camera-scale", type=float, default=0.08)
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--show-camera-images", dest="show_camera_images", action="store_true",
                             help="Attach selected camera frames to frustums (off by default for playback speed).")
    image_group.add_argument("--no-camera-images", dest="show_camera_images", action="store_false",
                             help=argparse.SUPPRESS)  # Backward-compatible spelling.
    parser.set_defaults(show_camera_images=False)
    parser.add_argument("--coordinate-frame", choices=("robot", "world"), default="robot",
                        help="Scene frame. robot uses inv(C2R) and is the default for Inspire captures.")
    parser.add_argument("--c2r", default=None,
                        help="world_from_robot C2R.npy. Default: <capture-dir>/C2R.npy when using robot coordinates.")
    parser.add_argument("--no-robot", action="store_true", help="Do not load or animate the recorded Inspire robot.")
    parser.add_argument("--robot-urdf", default=str(Path.home() / "paradex/rsc/robot/xarm_inspire_DFTP.urdf"),
                        help="Inspire arm+hand URDF used with raw/ arm and hand recordings.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cameras < 0 or args.frame_step < 1 or args.camera_image_max_side < 32 or args.camera_image_stride < 1:
        raise ValueError("camera and timeline parameters must be positive")
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    mesh_path = Path(args.object_mesh).expanduser().resolve()
    records_path = Path(args.gotrack_records).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else capture_dir / "undistorted_video"
    if not capture_dir.is_dir() or not mesh_path.is_file() or not records_path.is_file():
        raise FileNotFoundError("--capture-dir, --object-mesh, and --gotrack-records must exist")

    poses = _load_records(records_path)
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
        serials = list(args.camera_ids)
        missing = [serial for serial in serials if serial not in all_serials]
        if missing:
            raise ValueError(f"Unknown or uncalibrated camera IDs: {missing}")
    else:
        serials = all_serials if args.max_cameras == 0 else _evenly_spaced(all_serials, args.max_cameras)
    if not serials:
        raise ValueError("No calibrated cameras available")
    if args.show_camera_images:
        serials = [serial for serial in serials if _video_shape(video_dir / f"{serial}.avi") is not None]
    if not serials:
        raise ValueError("None of the selected cameras has a readable video")
    first_frame, last_frame = min(poses), max(poses)
    print(f"[viewer] frame={args.coordinate_frame} poses={len(poses)} frames={first_frame}..{last_frame} cameras={serials}")
    if args.dry_run:
        return 0

    try:
        import trimesh
        import viser
    except ModuleNotFoundError as exc:
        if exc.name == "viser":
            raise RuntimeError("Viser is not installed in this environment. Install it with: conda run -n gotrack pip install viser") from exc
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
    object_frame = server.scene.add_frame("/track/object", show_axes=True, axes_length=0.06, axes_radius=0.002)
    server.scene.add_mesh_trimesh("/track/object/mesh", mesh=mesh)

    readers = VideoFrames(video_dir, serials, args.camera_image_max_side) if args.show_camera_images else None
    camera_handles: dict[str, Any] = {}
    for serial in serials:
        # FOV/aspect must always use the real calibrated image dimensions.
        # A previous no-image shortcut used 1x1 here, degenerating every
        # frustum into a line instead of the expected rectangular pyramid.
        video_size = _video_shape(video_dir / f"{serial}.avi")
        if video_size is None:
            continue
        width, height = video_size
        camera_from_world = extrinsics[serial]
        world_from_camera = np.linalg.inv(camera_from_world)
        view_from_camera = view_from_world @ world_from_camera
        fov = float(2.0 * np.arctan2(height / 2.0, intrinsics[serial][1, 1]))
        handle = server.scene.add_camera_frustum(
            f"/cameras/{serial}", fov=fov, aspect=float(width / height), scale=args.camera_scale,
            wxyz=_wxyz(view_from_camera[:3, :3]), position=tuple(view_from_camera[:3, 3]),
            color=(255, 190, 80), image=None,
        )
        camera_handles[serial] = handle

        @handle.on_click
        def _(_, wxyz=_wxyz(view_from_camera[:3, :3]), position=tuple(view_from_camera[:3, 3])):
            for client in server.get_clients().values():
                client.camera.wxyz = wxyz
                client.camera.position = position

    trajectory = _transform_points(np.asarray([poses[index][:3, 3] for index in sorted(poses)]), view_from_world).astype(np.float32)
    try:
        server.scene.add_point_cloud("/track/trajectory", points=trajectory, colors=(255, 170, 0), point_size=0.004)
    except Exception:
        pass

    robot_qpos = robot_vis = robot_base = None
    if not args.no_robot:
        urdf_path = Path(args.robot_urdf).expanduser().resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Robot URDF is missing: {urdf_path}")
        try:
            paradex_root = Path.home() / "paradex"
            if str(paradex_root) not in sys.path:
                sys.path.insert(0, str(paradex_root))
            from viser.extras import ViserUrdf
            robot_qpos = _load_robot_qpos(capture_dir, last_frame + 1)
            robot_base = server.scene.add_frame("/robot", show_axes=True, axes_length=0.12, axes_radius=0.003)
            if args.coordinate_frame == "world":
                if world_from_robot is None:
                    raise FileNotFoundError("--coordinate-frame world with robot needs --c2r or <capture-dir>/C2R.npy")
                robot_base.position = tuple(world_from_robot[:3, 3])
                robot_base.wxyz = _wxyz(world_from_robot[:3, :3])
            # ViserUrdf updates only the 12 joint transforms per frame.  Do
            # not stream a freshly tessellated 260k-vertex robot mesh every
            # frame: that was the source of the previous slow, static-looking
            # playback.
            robot_vis = ViserUrdf(server, urdf_path, root_node_name="/robot")
            robot_vis.update_cfg(robot_qpos[first_frame])
            print(f"[viewer] robot={urdf_path} qpos_frames={len(robot_qpos)}")
        except Exception as exc:
            raise RuntimeError("Could not load the recorded Inspire arm/hand; use --no-robot to inspect object only") from exc

    with server.gui.add_folder("GoTrack inspection"):
        frame_gui = server.gui.add_slider("Frame", min=first_frame, max=last_frame, step=args.frame_step, initial_value=first_frame)
        play_gui = server.gui.add_checkbox("Play", initial_value=True)
        rate_gui = server.gui.add_slider("Playback FPS", min=1, max=90, step=1, initial_value=30)
        image_gui = server.gui.add_checkbox("Show camera frames", initial_value=readers is not None)
        robot_gui = server.gui.add_checkbox("Show robot", initial_value=robot_vis is not None)
        text_gui = server.gui.add_text("Pose status", initial_value="", disabled=True)

    lock = threading.Lock()
    last_pose: np.ndarray | None = None
    last_camera_image_frame: int | None = None

    def update(frame_index: int) -> None:
        nonlocal last_pose, last_camera_image_frame
        with lock:
            pose = poses.get(frame_index, last_pose)
            if pose is None:
                earlier = [index for index in poses if index <= frame_index]
                pose = poses[max(earlier)] if earlier else poses[min(poses)]
            last_pose = pose
            view_pose = view_from_world @ pose
            object_frame.position = tuple(float(value) for value in view_pose[:3, 3])
            object_frame.wxyz = _wxyz(view_pose[:3, :3])
            text_gui.value = f"frame={frame_index}; source pose={'exact' if frame_index in poses else 'held'}"
            if robot_vis is not None and robot_qpos is not None:
                robot_vis.update_cfg(robot_qpos[min(frame_index, len(robot_qpos) - 1)])
                robot_vis.show_visual = robot_gui.value
            if readers is not None:
                refresh_images = (last_camera_image_frame is None or not image_gui.value
                                  or frame_index - last_camera_image_frame >= args.camera_image_stride
                                  or frame_index < last_camera_image_frame)
                if refresh_images:
                    for serial, handle in camera_handles.items():
                        handle.image = readers.read(serial, frame_index) if image_gui.value else None
                    last_camera_image_frame = frame_index

    @frame_gui.on_update
    def _(_event: Any) -> None:
        update(int(frame_gui.value))

    update(first_frame)

    def playback() -> None:
        while True:
            if play_gui.value:
                next_frame = int(frame_gui.value) + args.frame_step
                frame_gui.value = first_frame if next_frame > last_frame else next_frame
            time.sleep(1.0 / max(1, int(rate_gui.value)))

    threading.Thread(target=playback, daemon=True).start()
    print(f"[viewer] open http://localhost:{args.port} (or this host's IP:{args.port}); Ctrl+C to stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        if readers is not None:
            readers.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
