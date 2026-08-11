#!/usr/bin/env python3
"""Render all-view GoTrack-object reprojection as one grid MP4.

The capture directory is read-only.  One or more object meshes and GoTrack
trajectories are rendered over each RGB view.  ``--object-only`` (recommended
for human-grasping captures) avoids loading the robot entirely.  Without it,
the optional robot overlay is also loaded from ``raw/``.

Run inside the ``gotrack`` environment:
    conda run -n gotrack python -u src/process/render_robot_object_reprojection_grid.py \
      --capture-dir /path/to/apple/0 \
      --object-mesh /path/to/apple.obj --gotrack-records /path/to/world_pose_records.json \
      --object-only \
      --output /path/to/robot_object_grid.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from articulated.joint_schema import load_single_joint_spec
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
PARADEX_ROOT = Path.home() / "paradex"
if str(PARADEX_ROOT) not in sys.path:
    sys.path.insert(0, str(PARADEX_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.process.gotrack_capture import _eligible_cameras

URDF_PATH = (REPO_ROOT / "autodex" / "planner" / "src" / "curobo" / "content" /
             "assets" / "robot" / "inspire_description" / "xarm_inspire.urdf")


def inspire_action_to_qpos(action: np.ndarray) -> np.ndarray:
    """Paradex visualize_all.py conversion for raw Inspire controller values.

    The raw controller order is ``little, ring, middle, index, thumb_2,
    thumb_1`` (0=closed, 1000=open); the URDF order is thumb first.
    Keeping this small conversion local avoids importing Paradex's optional
    Pinocchio wrapper just to obtain this pure NumPy mapping.
    """
    action = np.asarray(action, dtype=np.float64)
    qpos = np.zeros_like(action)
    qpos[:, 0] = 1.15 * (1.0 - action[:, 5] / 1000.0)  # thumb yaw
    qpos[:, 1] = 0.55 * (1.0 - action[:, 4] / 1000.0)  # thumb pitch
    qpos[:, 2] = 1.60 * (1.0 - action[:, 3] / 1000.0)  # index
    qpos[:, 3] = 1.60 * (1.0 - action[:, 2] / 1000.0)  # middle
    qpos[:, 4] = 1.60 * (1.0 - action[:, 1] / 1000.0)  # ring
    qpos[:, 5] = 1.60 * (1.0 - action[:, 0] / 1000.0)  # little
    return qpos


def _as_4x4(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 3x4 or 4x4 transform, got {matrix.shape}")
    return matrix


def _load_calibration(capture_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    intr_raw = json.loads((capture_dir / "cam_param" / "intrinsics.json").read_text())
    extr_raw = json.loads((capture_dir / "cam_param" / "extrinsics.json").read_text())
    shared = sorted(set(intr_raw) & set(extr_raw))
    intrinsics = {serial: np.asarray(intr_raw[serial]["intrinsics_undistort"], dtype=np.float32).reshape(3, 3)
                  for serial in shared}
    extrinsics = {serial: _as_4x4(extr_raw[serial]) for serial in shared}
    return intrinsics, extrinsics


def _interpolate(times_src: np.ndarray, data: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    return np.column_stack([np.interp(times_dst, times_src, values[:, i]) for i in range(values.shape[1])])


def _load_robot_qpos(capture_dir: Path, frame_count: int, arm_time_offset: float) -> np.ndarray:
    raw = capture_dir / "raw"
    arm_time = np.asarray(np.load(raw / "arm" / "time.npy", allow_pickle=True), dtype=np.float64)
    arm_qpos = np.asarray(np.load(raw / "arm" / "position.npy", allow_pickle=True), dtype=np.float64)
    hand_time = np.asarray(np.load(raw / "hand" / "time.npy", allow_pickle=True), dtype=np.float64)
    hand_raw = np.asarray(np.load(raw / "hand" / "position.npy", allow_pickle=True), dtype=np.float64)
    if arm_qpos.ndim != 2 or arm_qpos.shape[1] != 6 or hand_raw.ndim != 2 or hand_raw.shape[1] != 6:
        raise ValueError(f"Expected 6-DOF arm and hand streams, got arm={arm_qpos.shape}, hand={hand_raw.shape}")

    # This is the same conversion/order as Paradex visualize_all.py.
    hand_qpos = inspire_action_to_qpos(_interpolate(hand_time, hand_raw, arm_time))
    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)
    arm_time = arm_time + float(arm_time_offset)
    frame_times = np.asarray(np.load(raw / "timestamps" / "timestamp.npy", allow_pickle=True), dtype=np.float64)
    if len(frame_times) < 2:
        raise ValueError("Need at least two camera timestamps for robot synchronization")
    if len(frame_times) < frame_count:
        dt = float(np.median(np.diff(frame_times)))
        extra = frame_times[-1] + dt * np.arange(1, frame_count - len(frame_times) + 1)
        frame_times = np.concatenate([frame_times, extra])
    return _interpolate(arm_time, full_qpos, frame_times[:frame_count])


def _load_object_poses(path: Path) -> tuple[dict[int, np.ndarray], int]:
    """Load GoTrack JSON records or legacy all_poses_world.npz trajectories."""
    last_recorded_frame = -1
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        poses = {}
        for key in archive.files:
            if not key.startswith("frame_"):
                continue
            try:
                frame_index = int(key.removeprefix("frame_"))
            except ValueError:
                continue
            pose = np.asarray(archive[key], dtype=np.float64)
            if pose.shape == (4, 4) and np.isfinite(pose).all():
                poses[frame_index] = pose
                last_recorded_frame = max(last_recorded_frame, frame_index)
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
        last_recorded_frame = max((int(item["frame_index"]) for item in records), default=-1)
        poses = {int(item["frame_index"]): np.asarray(item["pose_world"], dtype=np.float64)
                 for item in records if item.get("pose_world") is not None}
    if not poses:
        raise ValueError(f"No valid pose_world records in {path}")
    return poses, last_recorded_frame


def _joint_transform(axis: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    """Rotation of ``theta`` about the line through ``origin`` along ``axis``."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / float(np.linalg.norm(axis))
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    rotation = (np.eye(3) + np.sin(theta) * cross
                + (1.0 - np.cos(theta)) * (cross @ cross))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = origin - rotation @ origin
    return pose


def _load_joint_angles(path: Path) -> dict[int, float]:
    """Per-frame joint angles from a GoTrack record, empty for a rigid one."""
    if path.suffix.lower() == ".npz":
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    angles: dict[int, float] = {}
    for item in records:
        if not isinstance(item, dict) or "frame_index" not in item:
            continue
        if "theta_rad" in item:
            angles[int(item["frame_index"])] = float(item["theta_rad"])
        elif "theta_deg" in item:
            angles[int(item["frame_index"])] = float(np.radians(item["theta_deg"]))
    return angles


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh: {path}")
    return mesh


def _grid_layout(count: int) -> tuple[int, int]:
    columns = int(np.ceil(np.sqrt(count)))
    return columns, int(np.ceil(count / columns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--video-dir", default=None,
                        help="Default: <capture-dir>/undistorted_video. Must match intrinsics_undistort.")
    parser.add_argument(
        "--moving-mesh", action="append", default=None,
        help=(
            "Articulated objects only, repeated alongside --object-mesh: the part on "
            "the far side of the joint. It is drawn as an extra overlay whose pose is "
            "the body pose composed with the joint angle read from the same records, "
            "so it needs no trajectory of its own. Omit it and an articulated run "
            "still renders, with the lid drawn shut over an open one."
        ),
    )
    parser.add_argument(
        "--articulation-json", action="append", default=None,
        help="Joint file per --moving-mesh, with axis and origin in the mesh frame.",
    )
    parser.add_argument("--object-mesh", action="append", required=True,
                        help="Object mesh. Repeat together with --gotrack-records for multiple objects.")
    parser.add_argument("--gotrack-records", action="append", required=True,
                        help=("world_pose_records.json or all_poses_world.npz. "
                              "Order must match repeated --object-mesh."))
    parser.add_argument("--object-only", action="store_true",
                        help="Render object overlays only; do not load/render the robot.")
    parser.add_argument("--output", required=True, help="New output .mp4; never overwritten.")
    parser.add_argument("--camera-ids", nargs="*", default=None, help="Default: every full-length capture camera.")
    parser.add_argument("--exclude-cameras", nargs="*", default=[],
                        help="Optional camera IDs to exclude explicitly. Default: none.")
    parser.add_argument("--min-video-frames", type=int, default=100,
                        help="Reject videos shorter than this before rendering.")
    parser.add_argument("--max-video-duration-skew-sec", type=float, default=1.0,
                        help=("Reject videos whose frame_count/FPS duration differs from the valid-camera "
                              "median by more than this. 0 disables it."))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="Inclusive; -1 means all available frames.")
    parser.add_argument("--grid-scale", type=float, default=0.20, help="Per-view scale in the output grid.")
    parser.add_argument("--arm-time-offset", type=float, default=0.0,
                        help="Seconds added to arm timestamps. This capture's raw arm and GoTrack events align at 0.0.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    video_dir = (Path(args.video_dir).expanduser().resolve() if args.video_dir else
                 capture_dir / "undistorted_video")
    object_mesh_paths = [Path(path).expanduser().resolve() for path in args.object_mesh]
    records_paths = [Path(path).expanduser().resolve() for path in args.gotrack_records]
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    if len(object_mesh_paths) != len(records_paths):
        raise ValueError("Repeat --object-mesh and --gotrack-records the same number of times")
    if not capture_dir.is_dir() or not video_dir.is_dir() or not all(path.is_file() for path in object_mesh_paths + records_paths):
        raise FileNotFoundError("capture directory/video directory, every object mesh, and every GoTrack record must exist")
    if not args.object_only and not URDF_PATH.is_file():
        raise FileNotFoundError(f"Inspire arm/hand URDF not found: {URDF_PATH}")
    if not 0 < args.grid_scale <= 1:
        raise ValueError("--grid-scale must be in (0, 1]")
    if args.min_video_frames < 1 or args.max_video_duration_skew_sec < 0:
        raise ValueError("--min-video-frames must be positive and --max-video-duration-skew-sec non-negative")

    intrinsics, extrinsics = _load_calibration(capture_dir)
    excluded = set(args.exclude_cameras)
    eligible, video_timings, rejected_cameras, median_duration = _eligible_cameras(
        video_dir, args.min_video_frames, excluded, args.max_video_duration_skew_sec,
    )
    candidates = sorted(serial for serial in eligible if serial in intrinsics)
    serials = list(args.camera_ids) if args.camera_ids else candidates
    invalid = [serial for serial in serials if serial not in candidates]
    if invalid:
        raise ValueError(f"Requested camera IDs unavailable, excluded, or short: {invalid}")
    if not serials:
        raise RuntimeError("No usable cameras")
    loaded_trajectories = [_load_object_poses(path) for path in records_paths]
    object_poses = [poses for poses, _ in loaded_trajectories]

    # A moving part becomes one more overlay rather than a special case in the render
    # loop: its pose is the body pose already loaded, composed with that frame's joint
    # angle. Everything downstream -- renderers, colours, holding the last pose over a
    # gap -- then applies to it unchanged.
    moving_meshes = list(args.moving_mesh or [])
    joint_files = list(args.articulation_json or [])
    if moving_meshes:
        if len(joint_files) != len(moving_meshes):
            raise ValueError("Repeat --articulation-json once per --moving-mesh")
        for index, (moving_path, joint_path) in enumerate(zip(moving_meshes, joint_files)):
            joint = load_single_joint_spec(joint_path)
            if joint.joint_type != "revolute":
                raise ValueError(
                    "The reprojection renderer currently poses a moving mesh by angle")
            axis = joint.axis
            origin = joint.origin
            angles = _load_joint_angles(records_paths[index])
            if not angles:
                raise ValueError(
                    f"{records_paths[index]} carries no joint angles, so "
                    f"{moving_path} cannot be posed"
                )
            body = object_poses[index]
            held = 0.0
            posed: dict[int, np.ndarray] = {}
            for frame_index in sorted(body):
                held = angles.get(frame_index, held)
                posed[frame_index] = body[frame_index] @ _joint_transform(axis, origin, held)
            object_mesh_paths.append(Path(moving_path).expanduser().resolve())
            object_poses.append(posed)
    common_video_frames = min(int(video_timings[serial]["frames"]) for serial in serials)
    total_frames = min(common_video_frames, *(last_frame + 1 for _, last_frame in loaded_trajectories))
    start = max(0, args.start_frame)
    end = total_frames - 1 if args.end_frame < 0 else min(args.end_frame, total_frames - 1)
    if end < start:
        raise ValueError(f"Invalid frame interval {start}..{end} for {total_frames} frames")
    columns, rows = _grid_layout(len(serials))
    manifest = {"serials": serials, "frames": [start, end], "grid": [columns, rows],
                "video_dir": str(video_dir), "object_only": args.object_only,
                "objects": [str(path) for path in records_paths],
                "arm_time_offset": args.arm_time_offset, "grid_scale": args.grid_scale,
                "manual_excluded_cameras": sorted(excluded),
                "rejected_cameras": rejected_cameras,
                "median_video_duration_sec": median_duration,
                "min_video_frames": args.min_video_frames,
                "max_video_duration_skew_sec": args.max_video_duration_skew_sec}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    from src.visualization.overlay_object_video_single import ObjectOverlayRenderer
    robot = robot_qpos = link_names = scene_meshes = link_labels = c2r = None
    if not args.object_only:
        try:
            from paradex.visualization.robot import RobotModule
        except ModuleNotFoundError as exc:
            if exc.name == "yourdfpy":
                raise RuntimeError(
                    "Robot reprojection needs yourdfpy in the gotrack environment. "
                    "Install it with: conda run -n gotrack pip install yourdfpy"
                ) from exc
            raise
        from src.visualization.overlay_robot_video import RobotOverlayRenderer, _label_for_link
        robot_qpos = _load_robot_qpos(capture_dir, total_frames, args.arm_time_offset)
        robot = RobotModule(str(URDF_PATH))
        robot.update_cfg(robot_qpos[0, :robot.get_num_joints()])
        scene = robot.scene
        link_names = list(scene.geometry.keys())
        scene_meshes = [scene.geometry[name] for name in link_names]
        link_labels = {name: _label_for_link(name) for name in link_names}
        c2r = _as_4x4(np.load(capture_dir / "C2R.npy"))

    object_meshes = [_load_mesh(path) for path in object_mesh_paths]

    caps: dict[str, cv2.VideoCapture] = {}
    robot_renderers = {}
    object_renderers: dict[str, list[ObjectOverlayRenderer]] = {}
    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0)]
    for serial in serials:
        cap = cv2.VideoCapture(str(video_dir / f"{serial}.avi"))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for camera {serial}")
        caps[serial] = cap
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if not args.object_only:
            cam_from_robot = extrinsics[serial] @ c2r
            robot_renderers[serial] = RobotOverlayRenderer(
                scene_meshes, link_names, link_labels, {serial: {"intrinsics_undistort": intrinsics[serial]}},
                {serial: cam_from_robot[:3, :]}, h, w,
            )
        object_renderers[serial] = [
            ObjectOverlayRenderer(mesh, {serial: intrinsics[serial]}, {serial: extrinsics[serial]}, h, w,
                                  color_bgr=colors[index % len(colors)], alpha=0.50)
            for index, mesh in enumerate(object_meshes)
        ]

    first_cap = caps[serials[0]]
    full_w, full_h = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cell_w, cell_h = max(1, round(full_w * args.grid_scale)), max(1, round(full_h * args.grid_scale))
    grid_size = (cell_w * columns, cell_h * rows)
    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, grid_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    last_object_poses = [None] * len(object_poses)
    try:
        for frame_idx in range(end + 1):
            frames = {}
            for serial, cap in caps.items():
                ok, frame = cap.read()
                frames[serial] = frame if ok else np.zeros((full_h, full_w, 3), dtype=np.uint8)
            if frame_idx < start:
                continue
            if not args.object_only:
                robot.update_cfg(robot_qpos[frame_idx, :robot.get_num_joints()])
                link_poses = [robot.scene.graph.get(name)[0] for name in link_names]
            frame_object_poses = []
            for index, poses in enumerate(object_poses):
                pose = poses.get(frame_idx, last_object_poses[index])
                if pose is not None:
                    last_object_poses[index] = pose
                frame_object_poses.append(pose)
            grid = np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8)
            for index, serial in enumerate(serials):
                rendered = frames[serial]
                if not args.object_only:
                    rendered = robot_renderers[serial].render(link_poses, [rendered])[0]
                for renderer, object_pose in zip(object_renderers[serial], frame_object_poses):
                    if object_pose is not None:
                        rendered = renderer.render(object_pose, [rendered])[0]
                cell = cv2.resize(rendered, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                cv2.putText(cell, f"{serial}  f{frame_idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 255), 1, cv2.LINE_AA)
                row, col = divmod(index, columns)
                grid[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = cell
            writer.write(grid)
            if (frame_idx - start) % 25 == 0 or frame_idx == end:
                print(f"[grid] {frame_idx - start + 1}/{end - start + 1}", flush=True)
    finally:
        writer.release()
        for cap in caps.values():
            cap.release()
    print(f"[done] {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
