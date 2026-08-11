"""Shared loading and posing helpers for the articulated-pose probe.

Nothing here writes to ``~/object_tracking`` or to ``capture/eccv2026/v0``. The
probe reads the promoted articulation result, the object meshes, and one
episode's calibration, and writes only under this directory.

The articulation model comes from ``articulation_particulate/joint.json``. Both the
legacy blue-box ``measured`` record and the promoted single-joint ``joints[]`` schema
are normalised by :mod:`joint_schema` before meshes or limits are consumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from joint_schema import load_single_joint_spec

SHARED = Path.home() / "shared_data"
MESH_ROOT = SHARED / "mesh_new"
CAPTURE_ROOT = SHARED / "capture/eccv2026/v0"

DEFAULT_OBJECT = "blue_plastic_box"
DEFAULT_EPISODE = CAPTURE_ROOT / "allegro_v5/blue_plastic_box/1"

BODY, LID = 0, 1


@dataclass(frozen=True)
class Articulation:
    """Two rigid parts joined by one revolute DoF, in the object's mesh frame."""

    body: trimesh.Trimesh
    lid: trimesh.Trimesh
    axis: np.ndarray          # (3,) unit direction
    origin: np.ndarray        # (3,) a point on the axis
    theta_min: float          # radians; lower joint limit
    theta_max: float          # radians; the scanned open state
    part_names: tuple[str, str] = ("body", "lid")

    def posed(self, pose_body: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Vertices of (body, lid) in world, with the lid swung by ``theta``."""
        lid_world = pose_body @ self.joint_transform(theta)
        return (
            trimesh.transform_points(self.body.vertices, pose_body),
            trimesh.transform_points(self.lid.vertices, lid_world),
        )

    def joint_transform(self, theta: float) -> np.ndarray:
        """Lid-relative-to-body transform at angle ``theta``, in the mesh frame."""
        return trimesh.transformations.rotation_matrix(theta, self.axis, self.origin)


def load_articulation(object_name: str = DEFAULT_OBJECT) -> Articulation:
    root = MESH_ROOT / object_name / "articulation_particulate"
    spec = load_single_joint_spec(root / "joint.json")
    if spec.joint_type != "revolute":
        raise ValueError(
            f"The FoundationPose seed supports a revolute joint, got {spec.joint_type}")
    return Articulation(
        body=trimesh.load(spec.part_paths[0], force="mesh"),
        lid=trimesh.load(spec.part_paths[1], force="mesh"),
        axis=spec.axis,
        origin=spec.origin,
        theta_min=spec.theta_min,
        theta_max=spec.theta_max,
        part_names=spec.part_names,
    )


@dataclass(frozen=True)
class Camera:
    """One calibrated camera. ``extrinsic`` maps world points into this camera."""

    camera_id: str
    K: np.ndarray             # (3, 3)
    extrinsic: np.ndarray     # (4, 4), world -> camera
    width: int
    height: int

    def scaled(self, factor: float) -> "Camera":
        """Same camera at a lower render resolution."""
        if factor == 1.0:
            return self
        K = self.K.copy()
        K[:2, :] *= factor
        return Camera(
            camera_id=self.camera_id,
            K=K,
            extrinsic=self.extrinsic,
            width=int(round(self.width * factor)),
            height=int(round(self.height * factor)),
        )

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """World points -> pixel coordinates. Points behind the camera come back NaN."""
        cam = trimesh.transform_points(np.atleast_2d(points_world), self.extrinsic)
        z = cam[:, 2]
        uv = (self.K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        uv[z <= 1e-6] = np.nan
        return uv


def load_cameras(episode: Path = DEFAULT_EPISODE) -> dict[str, Camera]:
    """The episode's calibrated cameras, keyed by serial.

    ``extrinsics.json`` holds 3x4 world->camera matrices, matching the convention
    the tracker uses (``gotrack_tracker.py`` inverts ``T_world_from_cam`` before
    handing extrinsics to triangulation).
    """
    intrinsics = json.loads((episode / "cam_param/intrinsics.json").read_text(encoding="utf-8"))
    extrinsics = json.loads((episode / "cam_param/extrinsics.json").read_text(encoding="utf-8"))
    if set(intrinsics) != set(extrinsics):
        raise ValueError(f"Intrinsic/extrinsic camera mismatch in {episode}")

    cameras: dict[str, Camera] = {}
    for camera_id in sorted(intrinsics):
        entry = intrinsics[camera_id]
        E = np.eye(4)
        E[:3, :4] = np.asarray(extrinsics[camera_id], dtype=np.float64)
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            K=np.asarray(entry["intrinsics_undistort"], dtype=np.float64).reshape(3, 3),
            extrinsic=E,
            width=int(entry["width"]),
            height=int(entry["height"]),
        )
    return cameras


def reference_pose(episode: Path = DEFAULT_EPISODE, frame: int | None = None) -> np.ndarray:
    """A real object pose from this episode, so the synthetic scene sits where the
    object actually sat. Defaults to the middle frame."""
    with np.load(episode / "object_6d_pose.npz") as data:
        keys = sorted(data.files, key=lambda name: int(name.split("_")[1]))
        if frame is None:
            frame = len(keys) // 2
        return np.asarray(data[keys[frame]], dtype=np.float64)


def theta_grid(theta_min: float, theta_max: float, step_deg: float = 2.0) -> np.ndarray:
    """A bounded sweep which keeps the published zero pose when it is in range."""
    lower, upper = np.degrees([theta_min, theta_max])
    if lower > upper or step_deg <= 0:
        raise ValueError("theta_min <= theta_max and a positive step are required")
    if lower <= 0.0 <= upper:
        negative = -np.arange(step_deg, abs(lower) + 1.0e-6, step_deg)[::-1]
        positive = np.arange(0.0, upper + 1.0e-6, step_deg)
        degrees = np.concatenate([negative, positive])
    else:
        degrees = np.arange(lower, upper + 1.0e-6, step_deg)
    return np.radians(degrees)
