"""Normalise the legacy and generic single-joint articulation file formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SingleJointSpec:
    joint_type: str
    axis: np.ndarray
    origin: np.ndarray
    limits: tuple[float, float]
    part_names: tuple[str, str]
    part_paths: tuple[Path, Path]
    parent_id: int
    child_id: int

    @property
    def theta_min(self) -> float:
        if self.joint_type != "revolute":
            raise ValueError("theta_min is defined only for a revolute joint")
        return self.limits[0]

    @property
    def theta_max(self) -> float:
        if self.joint_type != "revolute":
            raise ValueError("theta_max is defined only for a revolute joint")
        return self.limits[1]


def _part_path(root: Path, value: Any) -> Path:
    raw = Path(str(value)).expanduser()
    if raw.suffix.lower() == ".obj":
        candidate = raw if raw.is_absolute() else root / raw
        if candidate.is_file():
            return candidate.resolve()
    name = str(value)
    candidates = (root / "parts" / name / f"{name}.obj", root / "parts" / f"{name}.obj")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve part {value!r} beside {root / 'joint.json'}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def _limits(joint: dict[str, Any], joint_type: str) -> tuple[float, float]:
    if joint_type == "revolute":
        values = joint.get("range_rad")
        if values is None and joint.get("range_deg") is not None:
            values = np.radians(joint["range_deg"])
    else:
        values = (joint.get("range") or joint.get("travel")
                  or joint.get("range_m"))
    if values is None or len(values) != 2:
        unit = "range_rad" if joint_type == "revolute" else "range/range_m"
        raise ValueError(f"Single {joint_type} joint needs a two-value {unit}")
    lower, upper = (float(values[0]), float(values[1]))
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError(f"Invalid joint range [{lower}, {upper}]")
    return lower, upper


def load_single_joint_spec(path: str | Path) -> SingleJointSpec:
    """Load one joint and order its meshes as static parent, moving child.

    Supported inputs are the new ``parts`` + ``joints[]`` mesh schema, the blue-box
    ``measured`` schema, and GoTrack's flat runtime schema. Multiple joints are
    rejected deliberately: the current tracker has one joint coordinate.
    """
    path = Path(path).expanduser().resolve()
    root = path.parent
    data = json.loads(path.read_text(encoding="utf-8"))

    if "joints" in data:
        joints = data["joints"]
        if not isinstance(joints, list) or len(joints) != 1:
            count = len(joints) if isinstance(joints, list) else "non-list"
            raise ValueError(f"{path}: expected exactly one joint, got {count}")
        joint = dict(joints[0])
        parts = data.get("parts")
        if not isinstance(parts, dict):
            raise ValueError(f"{path}: generic schema needs a part-id to name mapping")
        parent_id, child_id = int(joint["parent"]), int(joint["child"])
        try:
            parent_name, child_name = str(parts[str(parent_id)]), str(parts[str(child_id)])
        except KeyError as exc:
            raise ValueError(f"{path}: joint references an unknown part id {exc.args[0]}") from exc
        part_names = (parent_name, child_name)
        part_paths = (_part_path(root, parent_name), _part_path(root, child_name))
    else:
        joint = dict(data.get("measured", data))
        parent_id, child_id = 0, 1
        raw_parts = data.get("parts") or joint.get("parts")
        if raw_parts is None:
            raw_parts = ["body", "lid"]
        if isinstance(raw_parts, dict):
            raw_parts = [raw_parts["0"], raw_parts["1"]]
        if not isinstance(raw_parts, list) or len(raw_parts) != 2:
            raise ValueError(f"{path}: flat single-joint schema needs exactly two parts")
        part_names = tuple(Path(str(value)).stem for value in raw_parts)
        part_paths = tuple(_part_path(root, value) for value in raw_parts)

    joint_type = str(joint.get("type", joint.get("joint_type", "revolute"))).lower()
    if joint_type not in {"revolute", "prismatic"}:
        raise ValueError(f"{path}: unsupported joint type {joint_type!r}")
    axis = np.asarray(joint["axis"], dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1.0e-5):
        raise ValueError(f"{path}: joint axis is not unit length: |axis|={norm}")
    origin = np.asarray(joint.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return SingleJointSpec(
        joint_type=joint_type,
        axis=axis,
        origin=origin,
        limits=_limits(joint, joint_type),
        part_names=(str(part_names[0]), str(part_names[1])),
        part_paths=(Path(part_paths[0]), Path(part_paths[1])),
        parent_id=parent_id,
        child_id=child_id,
    )
