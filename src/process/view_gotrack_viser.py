#!/usr/bin/env python3
"""Inspect rigid and articulated GoTrack trajectories with the recorded robot.

The viewer keeps one or more object meshes in the calibrated world frame and
updates their poses from ``world_pose_records.json`` with a frame slider.  The
primary object may have a moving mesh driven by a revolute or prismatic joint.
Unimanual and bimanual Inspire recordings are both supported.  It also shows a
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


def _joint_transform(
    joint_type: str, axis: np.ndarray, origin: np.ndarray, value: float,
) -> np.ndarray:
    """Return the local transform for a revolute or prismatic joint."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / float(np.linalg.norm(axis))
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    pose = np.eye(4)
    if joint_type == "prismatic":
        pose[:3, 3] = float(value) * axis
        return pose
    if joint_type != "revolute":
        raise ValueError(f"Unsupported articulation type: {joint_type!r}")
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    rotation = (np.eye(3) + np.sin(value) * cross
                + (1.0 - np.cos(value)) * (cross @ cross))
    pose[:3, :3] = rotation
    pose[:3, 3] = origin - rotation @ origin
    return pose


def _load_records(path: Path) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    poses: dict[int, np.ndarray] = {}
    # Rigid records simply leave this table empty. Articulated records use the
    # unit-native joint_value (metres for prismatic, radians for revolute).
    joint_values: dict[int, float] = {}
    for row in payload:
        if not isinstance(row, dict) or row.get("pose_world") is None:
            continue
        try:
            frame_index = int(row["frame_index"])
            poses[frame_index] = _as_4x4(row["pose_world"])
        except (KeyError, TypeError, ValueError):
            continue
        if row.get("joint_value") is not None:
            joint_values[frame_index] = float(row["joint_value"])
        elif "theta_rad" in row:
            joint_values[frame_index] = float(row["theta_rad"])
        elif "theta_deg" in row:
            joint_values[frame_index] = float(np.radians(row["theta_deg"]))
    if not poses:
        raise ValueError(f"No valid pose_world entries in {path}")
    return poses, joint_values


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


def _scene_name(name: str) -> str:
    return "_".join(name.strip().split()).replace("/", "_") or "object"


def _inspire_hand_qpos(hand_raw: np.ndarray) -> np.ndarray:
    """Convert the six Inspire controller values to URDF actuated joints."""
    # Controller order: little, ring, middle, index, thumb_2, thumb_1.
    hand = np.empty_like(hand_raw, dtype=np.float64)
    hand[:, 0] = 1.15 * (1.0 - hand_raw[:, 5] / 1000.0)
    hand[:, 1] = 0.55 * (1.0 - hand_raw[:, 4] / 1000.0)
    hand[:, 2] = 1.60 * (1.0 - hand_raw[:, 3] / 1000.0)
    hand[:, 3] = 1.60 * (1.0 - hand_raw[:, 2] / 1000.0)
    hand[:, 4] = 1.60 * (1.0 - hand_raw[:, 1] / 1000.0)
    hand[:, 5] = 1.60 * (1.0 - hand_raw[:, 0] / 1000.0)
    return hand


def _strict_time_samples(
    times: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort timestamps and retain the final value at duplicate times."""
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    keep = np.r_[times[1:] != times[:-1], True]
    return times[keep], values[keep]


def _load_one_robot_qpos(
    raw: Path, arm_name: str, hand_name: str, frame_times: np.ndarray,
) -> np.ndarray:
    arm_time = np.asarray(
        np.load(raw / arm_name / "time.npy", allow_pickle=True), dtype=np.float64)
    arm_qpos = np.asarray(
        np.load(raw / arm_name / "position.npy", allow_pickle=True), dtype=np.float64)
    hand_time = np.asarray(
        np.load(raw / hand_name / "time.npy", allow_pickle=True), dtype=np.float64)
    hand_raw = np.asarray(
        np.load(raw / hand_name / "position.npy", allow_pickle=True), dtype=np.float64)
    if (arm_qpos.ndim != 2 or arm_qpos.shape[1] != 6
            or hand_raw.ndim != 2 or hand_raw.shape[1] != 6):
        raise ValueError(
            f"Expected 6-DoF arm and hand streams, got "
            f"arm={arm_qpos.shape}, hand={hand_raw.shape}")
    if len(arm_time) < 2 or len(hand_time) < 2:
        raise ValueError("Robot streams must have at least two timestamped entries")
    arm_time, arm_qpos = _strict_time_samples(arm_time, arm_qpos)
    hand_time, hand_raw = _strict_time_samples(hand_time, hand_raw)
    hand = _inspire_hand_qpos(hand_raw)
    arm_at_frames = np.column_stack([
        np.interp(frame_times, arm_time, arm_qpos[:, index]) for index in range(6)
    ])
    hand_at_frames = np.column_stack([
        np.interp(frame_times, hand_time, hand[:, index]) for index in range(6)
    ])
    return np.concatenate([arm_at_frames, hand_at_frames], axis=1)


def _load_robot_qpos(
    capture_dir: Path, frame_count: int, *, video_fps: float,
    robot_video_offset_sec: float,
) -> dict[str, np.ndarray]:
    """Interpolate unimanual or bimanual Inspire streams onto video frames."""
    raw = capture_dir / "raw"
    if (raw / "arm_left").is_dir() and (raw / "arm_right").is_dir():
        streams = {
            "left": ("arm_left", "hand_left"),
            "right": ("arm_right", "hand_right"),
        }
    else:
        streams = {"right": ("arm", "hand")}
    timestamp_path = raw / "timestamps" / "timestamp.npy"
    if timestamp_path.is_file():
        frame_times = np.asarray(
            np.load(timestamp_path, allow_pickle=True), dtype=np.float64).reshape(-1)
        if len(frame_times) < 2:
            raise ValueError("Camera timestamp stream must have at least two entries")
        if len(frame_times) < frame_count:
            dt = float(np.median(np.diff(frame_times)))
            frame_times = np.r_[
                frame_times,
                frame_times[-1] + dt * np.arange(1, frame_count - len(frame_times) + 1),
            ]
        frame_times = frame_times[:frame_count] + float(robot_video_offset_sec)
    else:
        starts = [
            float(np.asarray(
                np.load(raw / arm_name / "time.npy", allow_pickle=True),
                dtype=np.float64)[0])
            for arm_name, _ in streams.values()
        ]
        frame_times = (
            min(starts) + float(robot_video_offset_sec)
            + np.arange(frame_count) / float(video_fps)
        )
    return {
        side: _load_one_robot_qpos(raw, arm_name, hand_name, frame_times)
        for side, (arm_name, hand_name) in streams.items()
    }


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
    parser.add_argument("--object-mesh", required=True,
                        help="For an articulated object this is the static part.")
    parser.add_argument(
        "--moving-mesh", default=None,
        help=(
            "Articulated objects only: the mesh on the far side of the joint, drawn "
            "as a child of the object frame so it picks up the body pose and then the "
            "joint angle. Without it an articulated record still displays, but with "
            "the lid frozen shut -- the body tracks correctly and the error is easy "
            "to miss, which is why the angle is also printed in the readout."
        ),
    )
    parser.add_argument(
        "--articulation-json", default=None,
        help="Joint file with axis and origin in the mesh frame, as written beside "
             "the part meshes. Required with --moving-mesh.",
    )
    parser.add_argument("--gotrack-records", required=True)
    parser.add_argument("--object-name", default=None,
                        help="Display name for the primary object; default: mesh stem.")
    parser.add_argument(
        "--additional-object", nargs=3, action="append", default=[],
        metavar=("NAME", "MESH", "GOTRACK_RECORDS"),
        help="Add another independently tracked rigid object. Repeatable.",
    )
    parser.add_argument("--video-dir", default=None, help="Default: <capture-dir>/undistorted_video")
    parser.add_argument("--camera-ids", nargs="*", default=None, help="Optional calibrated camera serials to show.")
    parser.add_argument("--max-cameras", type=int, default=0,
                        help="Evenly sample this many cameras when --camera-ids is omitted; 0 (default) shows all cameras.")
    parser.add_argument("--frame-step", type=int, default=1, help="Timeline slider step in tracked frames.")
    parser.add_argument("--camera-image-max-side", type=int, default=480)
    parser.add_argument("--camera-image-stride", type=int, default=10,
                        help="Refresh camera-frustum images every N playback frames (object/robot still update every frame).")
    parser.add_argument("--camera-scale", type=float, default=0.08)
    parser.add_argument("--material-roughness", type=float, default=1.0,
                        help="Viewer-only mesh roughness in [0, 1].")
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
                        help="Right/unimanual Inspire arm+hand URDF.")
    parser.add_argument("--left-robot-urdf", default=str(Path.home() / "paradex/rsc/robot/xarm_inspire_left_new.urdf"),
                        help="Left Inspire arm+hand URDF for bimanual recordings.")
    parser.add_argument("--right-c2r", default=None,
                        help="world_from_right_robot transform; defaults to --c2r.")
    parser.add_argument("--left-c2r", default=None,
                        help="world_from_left_robot transform for bimanual recordings.")
    parser.add_argument("--left-wrist-joint-offset-deg", type=float, default=0.0,
                        help="Viewer-only offset applied to left xArm joint6.")
    parser.add_argument("--video-fps", type=float, default=30.0,
                        help="Video rate when raw camera timestamps are absent.")
    parser.add_argument("--robot-video-offset-sec", type=float, default=0.0,
                        help="Robot elapsed time corresponding to video frame 0.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.max_cameras < 0 or args.frame_step < 1
            or args.camera_image_max_side < 32 or args.camera_image_stride < 1
            or args.video_fps <= 0 or not 0.0 <= args.material_roughness <= 1.0):
        raise ValueError(
            "camera/timeline parameters must be positive and roughness in [0, 1]")
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    primary_mesh_path = Path(args.object_mesh).expanduser().resolve()
    primary_records_path = Path(args.gotrack_records).expanduser().resolve()
    primary_name = args.object_name or primary_mesh_path.stem
    object_inputs = [(primary_name, primary_mesh_path, primary_records_path)]
    object_inputs.extend(
        (name, Path(mesh).expanduser().resolve(), Path(records).expanduser().resolve())
        for name, mesh, records in args.additional_object
    )
    object_names = [name for name, _, _ in object_inputs]
    if len(set(object_names)) != len(object_names):
        raise ValueError(f"Object display names must be unique: {object_names}")
    video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else capture_dir / "undistorted_video"
    missing_inputs = [
        str(path) for _, mesh, records in object_inputs
        for path in (mesh, records) if not path.is_file()
    ]
    if not capture_dir.is_dir() or missing_inputs:
        raise FileNotFoundError(
            "--capture-dir and every object mesh/record file must exist; "
            f"missing={missing_inputs}")

    loaded = {name: _load_records(records) for name, _, records in object_inputs}
    poses_by_object = {name: values[0] for name, values in loaded.items()}
    joints_by_object = {name: values[1] for name, values in loaded.items()}
    intrinsics, extrinsics = _load_calibration(capture_dir)
    c2r_path = Path(args.c2r).expanduser().resolve() if args.c2r else capture_dir / "C2R.npy"
    right_c2r_path = Path(args.right_c2r).expanduser().resolve() if args.right_c2r else c2r_path
    left_c2r_path = Path(args.left_c2r).expanduser().resolve() if args.left_c2r else None
    world_from_view_robot = _as_4x4(np.load(c2r_path)) if c2r_path.is_file() else None
    world_from_right = _as_4x4(np.load(right_c2r_path)) if right_c2r_path.is_file() else None
    world_from_left = (
        _as_4x4(np.load(left_c2r_path))
        if left_c2r_path is not None and left_c2r_path.is_file() else None
    )
    if args.coordinate_frame == "robot":
        if world_from_view_robot is None:
            raise FileNotFoundError(f"Robot-coordinate view requires C2R.npy: {c2r_path}")
        view_from_world = np.linalg.inv(world_from_view_robot)
    else:
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
    first_frame = min(min(poses) for poses in poses_by_object.values())
    last_frame = max(max(poses) for poses in poses_by_object.values())
    pose_counts = {name: len(poses) for name, poses in poses_by_object.items()}
    print(
        f"[viewer] frame={args.coordinate_frame} poses={pose_counts} "
        f"frames={first_frame}..{last_frame} cameras={serials}")
    if args.dry_run:
        return 0

    try:
        import trimesh
        import viser
    except ModuleNotFoundError as exc:
        if exc.name == "viser":
            raise RuntimeError("Viser is not installed in this environment. Install it with: conda run -n gotrack pip install viser") from exc
        raise

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", show_axes=True, axes_length=0.12, axes_radius=0.003)
    try:
        server.scene.add_grid("/world/grid", width=2.0, height=2.0, cell_size=0.1, section_size=0.5)
    except Exception:
        pass
    object_frames: dict[str, Any] = {}
    object_axis_frames: dict[str, Any] = {}
    for name, mesh_path, _ in object_inputs:
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load a Trimesh from {mesh_path}")
        material = getattr(mesh.visual, "material", None)
        if material is not None:
            try:
                material = material.to_pbr() if hasattr(material, "to_pbr") else material
                material.roughnessFactor = args.material_roughness
                material.metallicFactor = 0.0
                mesh.visual.material = material
            except (AttributeError, TypeError, ValueError):
                pass
        node = _scene_name(name)
        object_frames[name] = server.scene.add_frame(
            f"/track/{node}", show_axes=False)
        object_axis_frames[name] = server.scene.add_frame(
            f"/object_axes/{node}", show_axes=True,
            axes_length=0.06, axes_radius=0.002)
        server.scene.add_mesh_trimesh(f"/track/{node}/mesh", mesh=mesh)

    # The moving part hangs off the object frame, so viser composes the body pose
    # with the joint transform for us and the update loop only has to set one local
    # transform per frame.
    moving_frame = None
    joint_axis = joint_origin = None
    joint_type = None
    if args.moving_mesh:
        if not args.articulation_json:
            raise ValueError("--moving-mesh requires --articulation-json")
        joint = json.loads(
            Path(args.articulation_json).expanduser().read_text(encoding="utf-8"))
        joint = joint.get("measured", joint)
        if "joints" in joint:
            if len(joint["joints"]) != 1:
                raise ValueError("The viewer currently supports one articulated joint")
            joint = joint["joints"][0]
        joint_type = str(joint.get("joint_type", joint.get("type", "revolute")))
        joint_axis = np.asarray(joint["axis"], dtype=np.float64)
        joint_origin = np.asarray(joint.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        lid_mesh = trimesh.load(Path(args.moving_mesh).expanduser(), process=False)
        if isinstance(lid_mesh, trimesh.Scene):
            lid_mesh = trimesh.util.concatenate(tuple(lid_mesh.geometry.values()))
        primary_node = _scene_name(primary_name)
        moving_frame = server.scene.add_frame(
            f"/track/{primary_node}/moving", show_axes=False)
        server.scene.add_mesh_trimesh(
            f"/track/{primary_node}/moving/mesh", mesh=lid_mesh)

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

    trajectory_handles: dict[str, Any] = {}
    trajectory_colors = [
        (255, 170, 0), (60, 190, 255), (220, 80, 220), (80, 220, 120),
    ]
    for object_index, (name, poses) in enumerate(poses_by_object.items()):
        trajectory = _transform_points(
            np.asarray([poses[index][:3, 3] for index in sorted(poses)]),
            view_from_world,
        ).astype(np.float32)
        try:
            trajectory_handles[name] = server.scene.add_point_cloud(
                f"/trajectories/{_scene_name(name)}", points=trajectory,
                colors=trajectory_colors[object_index % len(trajectory_colors)],
                point_size=0.004,
            )
        except Exception:
            pass

    robot_qpos: dict[str, np.ndarray] = {}
    robot_visualizers: dict[str, Any] = {}
    if not args.no_robot:
        try:
            paradex_root = Path.home() / "paradex"
            if str(paradex_root) not in sys.path:
                sys.path.insert(0, str(paradex_root))
            from viser.extras import ViserUrdf
            robot_qpos = _load_robot_qpos(
                capture_dir, last_frame + 1, video_fps=args.video_fps,
                robot_video_offset_sec=args.robot_video_offset_sec,
            )
            if "left" in robot_qpos and args.left_wrist_joint_offset_deg:
                robot_qpos["left"] = robot_qpos["left"].copy()
                robot_qpos["left"][:, 5] += np.deg2rad(
                    args.left_wrist_joint_offset_deg)
            side_specs = {
                "right": (Path(args.robot_urdf).expanduser().resolve(), world_from_right),
                "left": (Path(args.left_robot_urdf).expanduser().resolve(), world_from_left),
            }
            for side, qpos in robot_qpos.items():
                urdf_path, world_from_side = side_specs[side]
                if not urdf_path.is_file():
                    raise FileNotFoundError(f"{side} robot URDF is missing: {urdf_path}")
                if world_from_side is None:
                    print(
                        f"[viewer] warning: omitting {side} robot because its C2R "
                        "is unavailable")
                    continue
                view_from_side = view_from_world @ world_from_side
                server.scene.add_frame(
                    f"/robots/{side}", show_axes=True,
                    axes_length=0.12, axes_radius=0.003,
                    position=tuple(view_from_side[:3, 3]),
                    wxyz=_wxyz(view_from_side[:3, :3]),
                )
                visualizer = ViserUrdf(
                    server, urdf_path, root_node_name=f"/robots/{side}")
                visualizer.update_cfg(qpos[first_frame])
                robot_visualizers[side] = visualizer
                print(
                    f"[viewer] robot[{side}]={urdf_path} qpos_frames={len(qpos)}")
            if not robot_visualizers:
                raise ValueError("No robot could be placed; provide valid C2R paths")
        except Exception as exc:
            raise RuntimeError("Could not load the recorded Inspire arm/hand; use --no-robot to inspect object only") from exc

    with server.gui.add_folder("GoTrack inspection"):
        frame_gui = server.gui.add_slider("Frame", min=first_frame, max=last_frame, step=args.frame_step, initial_value=first_frame)
        play_gui = server.gui.add_checkbox("Play", initial_value=True)
        rate_gui = server.gui.add_slider("Playback FPS", min=1, max=90, step=1, initial_value=30)
        image_gui = server.gui.add_checkbox("Show camera frames", initial_value=readers is not None)
        trajectory_gui = server.gui.add_checkbox("Show trajectories", initial_value=True)
        object_axes_gui = server.gui.add_checkbox("Show object axes", initial_value=True)
        robot_gui = server.gui.add_checkbox("Show robot", initial_value=bool(robot_visualizers))
        text_gui = server.gui.add_text("Pose status", initial_value="", disabled=True)

    lock = threading.Lock()
    last_pose: dict[str, np.ndarray] = {}
    last_camera_image_frame: int | None = None

    def update(frame_index: int) -> None:
        nonlocal last_camera_image_frame
        with lock:
            pose_status = []
            for name, poses in poses_by_object.items():
                pose = poses.get(frame_index, last_pose.get(name))
                if pose is None:
                    earlier = [index for index in poses if index <= frame_index]
                    pose = poses[max(earlier)] if earlier else poses[min(poses)]
                last_pose[name] = pose
                view_pose = view_from_world @ pose
                object_frame = object_frames[name]
                object_frame.position = tuple(float(value) for value in view_pose[:3, 3])
                object_frame.wxyz = _wxyz(view_pose[:3, :3])
                axis_frame = object_axis_frames[name]
                axis_frame.position = object_frame.position
                axis_frame.wxyz = object_frame.wxyz
                pose_status.append(
                    f"{name}={'exact' if frame_index in poses else 'held'}")

            primary_joints = joints_by_object[primary_name]
            if primary_joints:
                exact = frame_index in primary_joints
                earlier = [index for index in primary_joints if index <= frame_index]
                value = primary_joints.get(
                    frame_index,
                    primary_joints[max(earlier)] if earlier
                    else primary_joints[min(primary_joints)],
                )
                if moving_frame is not None:
                    local = _joint_transform(
                        joint_type, joint_axis, joint_origin, value)
                    moving_frame.position = tuple(float(v) for v in local[:3, 3])
                    moving_frame.wxyz = _wxyz(local[:3, :3])
                display = (
                    f"{value * 1000.0:.1f} mm" if joint_type == "prismatic"
                    else f"{np.degrees(value):.1f} deg"
                )
                pose_status.append(
                    f"{primary_name}.joint={display} "
                    f"({'exact' if exact else 'held'})")
            text_gui.value = f"frame={frame_index}; " + ", ".join(pose_status)

            for side, robot_vis in robot_visualizers.items():
                qpos = robot_qpos[side]
                robot_vis.update_cfg(qpos[min(frame_index, len(qpos) - 1)])
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

    @trajectory_gui.on_update
    def _(_event: Any) -> None:
        for handle in trajectory_handles.values():
            handle.visible = bool(trajectory_gui.value)

    @object_axes_gui.on_update
    def _(_event: Any) -> None:
        for handle in object_axis_frames.values():
            handle.visible = bool(object_axes_gui.value)

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
