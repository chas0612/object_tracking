#!/usr/bin/env python3
"""Turn a Particulate prediction into the joint file and part meshes the tracker takes.

Particulate predicts in a frame of its own making: the mesh is rotated to a canonical
up direction and then normalised into a [-0.5, 0.5]^3 box before anything is inferred
(``infer.py``, ``predict_mesh``). Its ``pred.npz`` carries the prediction and *not*
the transform that produced it, so undoing it is the caller's job and has to be done
by recomputing it from the same input mesh.

Which quantities that touches is not uniform, and that is the whole reason this file
exists rather than a few lines in a notebook:

======================  =====================================  ===================
quantity                undo                                   silent if wrong?
======================  =====================================  ===================
axis (either type)      rotate by R^T                           points the wrong way
revolute origin         rotate, scale, translate                hinge in the wrong place
``revolute_range``      nothing -- radians are dimensionless    n/a
``prismatic_range``     **multiply by the mesh's bbox extent**  travel off by ~1/size
======================  =====================================  ===================

The last row is the one to watch. A revolute range needed no conversion at all, which
is why nothing in this pipeline has ever had to think about it; a prismatic range is a
length and needs the same scaling the origin gets. Nothing raises when it is skipped --
the joint simply travels a factor of the object's size too far.

Run:

    python src/process/articulated/joint_from_particulate.py \\
        --pred out/<obj>/pred.npz --mesh <obj>.obj --up-dir Z --out <dir>

    python src/process/articulated/joint_from_particulate.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

# The up-direction rotations Particulate applies before normalising, copied from
# ``infer.py``. Same matrices, keyed the same way, because the inverse has to be the
# inverse of what actually ran -- not of what the docstring there says.
UP_DIR_ROTATIONS = {
    "X": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    "-X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    "Y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    "-Y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
    "Z": np.eye(3, dtype=np.float64),
    "-Z": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64),
}


def normalisation_from_mesh(vertices: np.ndarray, up_dir: str) -> tuple[np.ndarray, np.ndarray, float]:
    """The rotation, centre and scale Particulate would have applied to this mesh.

    Recomputed rather than read back, because ``pred.npz`` does not store them. It is
    deterministic -- ``x_norm = (R x - centre) / scale`` -- but it depends on the
    ``--up_dir`` the inference actually ran with, which is also not stored. Passing a
    different one here does not raise; it silently rotates the joint.
    """
    if up_dir not in UP_DIR_ROTATIONS:
        raise ValueError(f"up_dir must be one of {sorted(UP_DIR_ROTATIONS)}, got {up_dir!r}")
    rotation = UP_DIR_ROTATIONS[up_dir]
    rotated = np.asarray(vertices, dtype=np.float64) @ rotation.T
    lower, upper = rotated.min(axis=0), rotated.max(axis=0)
    centre = (lower + upper) / 2.0
    scale = float((upper - lower).max())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Mesh has a degenerate bounding box")
    return rotation, centre, scale


def direction_to_mesh_frame(direction: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """A direction only rotates back. Uniform scaling cannot change it."""
    out = rotation.T @ np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(out))
    if norm < 1.0e-9:
        raise ValueError("Predicted axis is degenerate")
    return out / norm


def point_to_mesh_frame(point: np.ndarray, rotation: np.ndarray,
                        centre: np.ndarray, scale: float) -> np.ndarray:
    """A point takes the whole inverse: undo the normalisation, then the rotation."""
    return rotation.T @ (np.asarray(point, dtype=np.float64).reshape(3) * scale + centre)


def plucker_to_axis_point(plucker: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Particulate's own convention, reproduced so the two cannot drift apart."""
    line, moment = np.asarray(plucker, dtype=np.float64)[:3], np.asarray(plucker, dtype=np.float64)[3:]
    axis = line / (np.linalg.norm(line) + 1.0e-8)
    return axis, np.cross(moment, axis)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return mesh


def joint_from_prediction(prediction: dict, rotation: np.ndarray, centre: np.ndarray,
                          scale: float, moving_part: int | None = None) -> dict:
    """The joint, in the input mesh's own frame and units.

    Refuses more than one moving part and refuses a part predicted as both revolute
    and prismatic. Neither is a limitation of this converter -- both are statements
    that the object does not have the one degree of freedom the tracker models, and
    quietly picking one of the two would move the discarded motion into the body pose,
    where it comes back as a body that drifts instead of a joint that is wrong.
    """
    is_revolute = np.asarray(prediction["is_part_revolute"], dtype=bool).reshape(-1)
    is_prismatic = np.asarray(prediction["is_part_prismatic"], dtype=bool).reshape(-1)
    moving = np.flatnonzero(is_revolute | is_prismatic)
    if moving_part is not None:
        if moving_part not in moving.tolist():
            raise ValueError(
                f"part {moving_part} is predicted as fixed; moving parts are {moving.tolist()}")
        moving = np.array([moving_part])
    if moving.size == 0:
        raise ValueError("Particulate predicted no moving part; the object is rigid to it")
    if moving.size > 1:
        raise ValueError(
            f"Particulate predicted {moving.size} moving parts {moving.tolist()}; the "
            "tracker models exactly one joint. Pass --moving-part to choose, having "
            "decided which one the capture actually exercises")

    part = int(moving[0])
    if is_revolute[part] and is_prismatic[part]:
        # Particulate's motion classifier has a fourth class, "both", and its own
        # visualisation resolves it by preferring revolute. Do not copy that here: it
        # is the model saying this joint is not one degree of freedom, and the
        # discarded half does not disappear -- it is absorbed by the body pose.
        axis_r, _ = plucker_to_axis_point(prediction["revolute_plucker"][part])
        axis_p = np.asarray(prediction["prismatic_axis"][part], dtype=np.float64)
        alignment = abs(float(np.dot(axis_r / np.linalg.norm(axis_r),
                                     axis_p / np.linalg.norm(axis_p))))
        raise ValueError(
            f"part {part} is predicted as BOTH revolute and prismatic, which is two "
            f"joint degrees of freedom; the tracker models one. The two axes are "
            f"{np.degrees(np.arccos(np.clip(alignment, 0.0, 1.0))):.1f} deg apart "
            f"(|cos| = {alignment:.3f}). Parallel axes mean a cylindrical or screw "
            "joint -- decide which, and whether the two motions are independent or "
            "coupled by a thread, before tracking. See the README.")

    if is_prismatic[part]:
        axis = direction_to_mesh_frame(prediction["prismatic_axis"][part], rotation)
        low, high = (float(v) for v in np.asarray(prediction["prismatic_range"][part]).reshape(2))
        # The one conversion with no counterpart in the revolute path. Particulate
        # measured this inside a unit box; the tracker measures it in the mesh.
        joint = {
            "joint_type": "prismatic",
            "axis": axis.tolist(),
            "range": [low * scale, high * scale],
            "range_normalised": [low, high],
        }
    else:
        axis_n, point_n = plucker_to_axis_point(prediction["revolute_plucker"][part])
        joint = {
            "joint_type": "revolute",
            "axis": direction_to_mesh_frame(axis_n, rotation).tolist(),
            "origin": point_to_mesh_frame(point_n, rotation, centre, scale).tolist(),
            # Radians. Dimensionless, so the normalisation never touched it.
            "range_rad": [float(v) for v in np.asarray(prediction["revolute_range"][part]).reshape(2)],
        }
    joint["moving_part_id"] = part
    joint["normalisation"] = {"scale": scale, "centre": centre.tolist(), "up_dir_applied": True}
    return joint


def export_parts(mesh: trimesh.Trimesh, face_part_ids: np.ndarray, moving_part: int,
                 out_dir: Path) -> list[str]:
    """Body and moving part as separate meshes, in the input mesh's own units.

    Particulate's own export writes the *normalised* mesh. The tracker needs the
    original, and the face ordering is untouched by normalisation -- only vertices
    move -- so the predicted face labels index this mesh directly.

    Body first, moving part second: the anchor bank defines part 0 as static and part
    1 as the one beyond the joint by the order of the paths, and infers nothing.
    """
    labels = np.asarray(face_part_ids, dtype=np.int64).reshape(-1)
    if labels.shape[0] != int(np.asarray(mesh.faces).shape[0]):
        raise ValueError(
            f"pred.npz labels {labels.shape[0]} faces but the mesh has "
            f"{int(np.asarray(mesh.faces).shape[0])}; this prediction is not for this mesh")
    moving_mask = labels == int(moving_part)
    # Unassigned faces (-1 under strict part assignment) stay with the body: a face
    # the model could not place is not evidence that it moves.
    body_mask = ~moving_mask
    if not moving_mask.any() or not body_mask.any():
        raise ValueError("One of the two parts came out empty")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, mask in (("body", body_mask), ("moving", moving_mask)):
        path = out_dir / f"{name}.obj"
        mesh.submesh([mask], append=True).export(path)
        paths.append(str(path))
    return paths


def convert(pred_path: Path, mesh_path: Path, up_dir: str, out_dir: Path,
            moving_part: int | None = None) -> dict:
    mesh = _load_mesh(mesh_path)
    rotation, centre, scale = normalisation_from_mesh(mesh.vertices, up_dir)
    with np.load(pred_path, allow_pickle=False) as payload:
        prediction = {key: payload[key] for key in payload.files}
    joint = joint_from_prediction(prediction, rotation, centre, scale, moving_part)
    joint["parts"] = export_parts(
        mesh, prediction["face_part_ids"], joint["moving_part_id"], out_dir)
    joint["source"] = {"pred": str(pred_path), "mesh": str(mesh_path), "up_dir": up_dir}
    (out_dir / "joint.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    return joint


def self_test() -> int:
    """Round-trip a known box through the normalisation, with no model involved.

    There is no prismatic object here to test against, and this is what can be
    checked without one: that a travel written in mesh units comes back in mesh units.
    The box is deliberately not a cube and not centred, so a missing rotation, a
    missing recentring or a missing scale each move the answer.
    """
    failures = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not condition:
            failures.append(name)

    mesh = trimesh.creation.box(extents=(0.18, 0.30, 0.12))
    mesh.apply_translation([0.4, -0.25, 1.1])
    scale_true = 0.30
    rotation, centre, scale = normalisation_from_mesh(mesh.vertices, "Z")
    check("scale is the largest extent", abs(scale - scale_true) < 1e-12, f"{scale:.6f}")

    travel_mesh = 0.15
    prediction = {
        "is_part_revolute": np.array([False, False]),
        "is_part_prismatic": np.array([False, True]),
        "prismatic_axis": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        # What Particulate would have predicted for a 0.15 m travel on this mesh.
        "prismatic_range": np.array([[0.0, 0.0], [0.0, travel_mesh / scale_true]]),
        "revolute_plucker": np.zeros((2, 6)),
        "revolute_range": np.zeros((2, 2)),
    }
    joint = joint_from_prediction(prediction, rotation, centre, scale)
    check("travel comes back in mesh units",
          abs(joint["range"][1] - travel_mesh) < 1e-12, f"{joint['range'][1]:.6f} m")
    check("axis is a unit direction", abs(np.linalg.norm(joint["axis"]) - 1.0) < 1e-12)
    check("prismatic carries no origin", "origin" not in joint)

    # The same prediction read under the wrong up direction. The axis rotates; the
    # travel does *not*, because the up-direction rotations are axis permutations and
    # the largest bounding-box extent is invariant under those. So a wrong --up-dir is
    # invisible in every number this script prints except the axis itself, which is
    # why it has to be carried across by hand from the inference run and cannot be
    # inferred from a sanity check on the range.
    rotation_x, centre_x, scale_x = normalisation_from_mesh(mesh.vertices, "X")
    joint_x = joint_from_prediction(prediction, rotation_x, centre_x, scale_x)
    check("a wrong up_dir rotates the axis",
          not np.allclose(joint_x["axis"], joint["axis"]),
          f"{np.round(joint['axis'], 3).tolist()} -> {np.round(joint_x['axis'], 3).tolist()}")
    check("and leaves the travel unchanged, so it cannot be caught there",
          abs(joint_x["range"][1] - joint["range"][1]) < 1e-12)

    # A revolute range must survive untouched, or a future edit that "fixes" the units
    # by scaling every range uniformly would pass everything above.
    revolute = {
        "is_part_revolute": np.array([False, True]),
        "is_part_prismatic": np.array([False, False]),
        "prismatic_axis": np.zeros((2, 3)),
        "prismatic_range": np.zeros((2, 2)),
        "revolute_plucker": np.array([[0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0.2, 0]]),
        "revolute_range": np.array([[0.0, 0.0], [0.0, 3.6]]),
    }
    joint_r = joint_from_prediction(revolute, rotation, centre, scale)
    check("radian range is not scaled", abs(joint_r["range_rad"][1] - 3.6) < 1e-12,
          f"{joint_r['range_rad'][1]:.6f} rad")
    check("revolute origin is scaled and translated",
          np.linalg.norm(np.asarray(joint_r["origin"]) - centre) > 1e-6)

    both = dict(revolute)
    both["is_part_prismatic"] = np.array([False, True])
    both["prismatic_axis"] = np.array([[0.0, 0, 1.0], [1.0, 0.0, 0.0]])
    try:
        joint_from_prediction(both, rotation, centre, scale)
        check("a 'both' prediction is refused", False, "no error raised")
    except ValueError as error:
        check("a 'both' prediction is refused", "BOTH" in str(error))

    print(f"\n{len(failures)} failures" + (f": {failures}" if failures else ""))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=str, help="Particulate's pred.npz")
    parser.add_argument("--mesh", type=str, help="the mesh that prediction was run on")
    parser.add_argument("--up-dir", type=str, default="Z", choices=sorted(UP_DIR_ROTATIONS),
                        help="must match the --up_dir the inference ran with; not stored in pred.npz")
    parser.add_argument("--out", type=str, help="output directory for joint.json and the parts")
    parser.add_argument("--moving-part", type=int, default=None,
                        help="which predicted part is beyond the joint, when more than one moves")
    parser.add_argument("--self-test", action="store_true",
                        help="round-trip the unit conversions on a synthetic box and exit")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not (args.pred and args.mesh and args.out):
        parser.error("--pred, --mesh and --out are required unless --self-test is given")

    joint = convert(Path(args.pred).expanduser(), Path(args.mesh).expanduser(),
                    args.up_dir, Path(args.out).expanduser(), args.moving_part)
    print(json.dumps({key: value for key, value in joint.items()
                      if key != "normalisation"}, indent=2))
    if joint["joint_type"] == "prismatic":
        low, high = joint["range"]
        print(f"\ntravel {high - low:.4f} mesh units "
              f"(normalised {joint['range_normalised'][1] - joint['range_normalised'][0]:.4f} "
              f"x scale {joint['normalisation']['scale']:.4f})")
        print("Treat the range as a guard, not a measurement: it is how far the scan "
              "happened to be opened.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
