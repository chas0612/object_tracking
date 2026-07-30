#!/usr/bin/env python3
"""FoundationPose control: per-part rigid pose from the one RGB-D view we have.

Runs FoundationPose twice per frame -- once on ``body.obj``, once on ``lid.obj``
-- then recovers the lid angle by projecting the measured body-to-lid transform
onto the joint's one-parameter family. This is the alternative to
``fit_rc.py`` for supplying the tracker's initial 7-DoF pose, so the two are
scored on identical frames.

Runs in the **object_6d** env, against ``~/object-6d-tracking``'s FoundationPose,
which is the installation that actually has weights and a built ``mycpp``. The
copy vendored under ``~/object_tracking`` has neither.

Two ways this probe is deliberately *generous* to FoundationPose, both worth
remembering when reading the numbers:

* It is handed a **per-part** mask (body pixels for the body run, lid pixels for
  the lid run). At run time SAM3 segments the object, not its parts, so this
  information would not exist. ``fit_rc.py`` by contrast gets only the union.
* It gets depth, which ``fit_rc.py`` never touches.

And one way the fixture is unkind to it: the synthetic RGB is flat Lambertian
shading with no texture, so its appearance cues are weaker here than they would
be on real footage. Depth still carries the geometry, which is the part this
probe is really asking about -- whether a thin flat lid can be localised alone.

GPU use is capped at ``--budget-gib`` so it cannot starve the tracking campaign
sharing this card.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import trimesh

# Overridable so the vendored copy under autodex/perception/thirdparty can be
# used instead of the standalone checkout. Same code, but the vendored one is
# tracked in git and carries the OOM and pytorch3d fixes this machine needs.
REPO_ROOT = Path(__file__).resolve().parents[3]
# The vendored copy is the default: it is tracked in git and carries the reduced
# predictor batch size and the guarded pytorch3d imports this machine needs. The
# environment variable still points at a standalone checkout when one is preferred.
FP_ROOT = Path(os.environ.get(
    "FOUNDATIONPOSE_ROOT",
    REPO_ROOT / "autodex/perception/thirdparty/FoundationPose"))

from common import BODY, LID, load_articulation, load_cameras  # noqa: E402


def _decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce a raw scan to something FoundationPose can batch.

    FoundationPose renders all 252 pose hypotheses in one nvdiffrast call, so its
    peak memory scales with hypotheses x vertices. The raw Artec parts are 240k
    and 86k faces, which overruns any sane budget on a shared card. Decimating is
    not a handicap -- it is how FoundationPose is normally fed, and the earlier
    Particulate work measured the full-vs-decimated difference on this same
    object as smaller than run-to-run noise.
    """
    if len(mesh.faces) <= target_faces:
        return mesh
    import open3d as o3d

    source = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    reduced = source.simplify_quadric_decimation(int(target_faces))
    reduced.compute_vertex_normals()
    out = trimesh.Trimesh(vertices=np.asarray(reduced.vertices),
                          faces=np.asarray(reduced.triangles), process=False)
    out.vertex_normals = np.asarray(reduced.vertex_normals)
    return out


def _joint_angle(delta: np.ndarray, axis: np.ndarray, origin: np.ndarray) -> tuple[float, float]:
    """Best theta explaining ``delta`` as a rotation about the joint, and the
    residual left over.

    The residual is the honesty check: if the two independent FoundationPose fits
    really do describe a hinge, ``delta`` lies close to the one-parameter family
    and this is small. A large residual means the pair is inconsistent, whatever
    the angle says.
    """
    best = (np.inf, 0.0)
    for theta in np.radians(np.arange(0.0, 360.0, 0.25)):
        model = trimesh.transformations.rotation_matrix(theta, axis, origin)
        error = float(np.linalg.norm(delta - model))
        if error < best[0]:
            best = (error, float(theta))
    return best[1], best[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=Path(__file__).parent / "synthetic")
    parser.add_argument("--budget-gib", type=float, default=6.0)
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--decimate-faces", type=int, default=20000,
                        help="Face budget per part mesh. 0 keeps the raw scan.")
    parser.add_argument("--rotation-views", type=int, default=12,
                        help="Viewpoint hypotheses. Upstream uses 40 (-> 252 poses with "
                             "the inplane step), and renders them in ONE nvdiffrast batch "
                             "-- peak memory is linear in this. Lowered so the probe fits "
                             "beside the tracking campaign; raise it if accuracy looks "
                             "hypothesis-starved.")
    parser.add_argument("--inplane-step", type=float, default=60.0)
    args = parser.parse_args()

    import torch
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.budget_gib / total_gib))

    sys.path.insert(0, str(FP_ROOT))
    sys.path.insert(0, str(FP_ROOT / "mycpp/build"))
    import nvdiffrast.torch as dr
    import estimater as estimater_module
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    # ``estimater`` swallows a failed ``import mycpp`` at module load and leaves the
    # name as None, then calls ``mycpp.cluster_poses`` anyway. Its own import runs
    # before our sys.path entry exists, so inject the module by hand -- the same
    # fix autodex/perception/pose.py applies.
    if getattr(estimater_module, "mycpp", None) is None:
        import mycpp
        estimater_module.mycpp = mycpp

    truth = json.loads((args.synthetic / "ground_truth.json").read_text(encoding="utf-8"))
    articulation = load_articulation(truth["object"])
    cameras = {cid: cam.scaled(truth["render_scale"])
               for cid, cam in load_cameras(Path(truth["episode"])).items()}
    pose_true = np.asarray(truth["pose_body"], dtype=np.float64)
    camera_id = truth["depth_camera_ids"][0]
    camera = cameras[camera_id]

    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
    estimators = {}
    for part, raw in ((BODY, articulation.body), (LID, articulation.lid)):
        mesh = _decimate(raw, args.decimate_faces) if args.decimate_faces else raw
        print(f"  part {part}: {len(raw.faces)} -> {len(mesh.faces)} faces", flush=True)
        estimator = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=scorer, refiner=refiner, glctx=glctx,
            debug=0, debug_dir="/tmp/fpose_probe")
        estimator.make_rotation_grid(min_n_views=args.rotation_views,
                                     inplane_step=args.inplane_step)
        print(f"    hypotheses: {len(estimator.rot_grid)}", flush=True)
        estimators[part] = estimator

    print(f"camera={camera_id}  {camera.width}x{camera.height}  "
          f"budget={args.budget_gib} GiB  refine_iters={args.refine_iterations}", flush=True)
    print(f"{'theta_gt':>9} {'theta_err':>10} {'body_rot':>9} {'body_trans':>11} "
          f"{'hinge_resid':>12} {'sec':>6}", flush=True)

    rows = []
    for frame in truth["frames"]:
        frame_dir = args.synthetic / f"frame_{frame['frame']:03d}"
        parts = np.load(frame_dir / f"{camera_id}_parts.npy")
        rgb = np.load(frame_dir / f"{camera_id}_rgb.npy")
        depth = np.load(frame_dir / f"{camera_id}_depth.npy")

        started = time.perf_counter()
        poses_world = {}
        for part, estimator in estimators.items():
            mask = parts == part
            if mask.sum() < 200:
                poses_world[part] = None
                continue
            in_camera = estimator.register(
                K=camera.K.astype(np.float64), rgb=rgb, depth=depth.astype(np.float32),
                ob_mask=mask, iteration=args.refine_iterations)
            poses_world[part] = np.linalg.inv(camera.extrinsic) @ np.asarray(in_camera)
        elapsed = time.perf_counter() - started

        if poses_world[BODY] is None or poses_world[LID] is None:
            print(f"{frame['theta_deg']:9.1f}   part not visible enough; skipped", flush=True)
            continue

        body_pose = poses_world[BODY]
        delta = np.linalg.inv(body_pose) @ poses_world[LID]
        theta, residual = _joint_angle(delta, articulation.axis, articulation.origin)

        offset = np.linalg.inv(pose_true) @ body_pose
        rot_err = np.degrees(np.arccos(
            np.clip((np.trace(offset[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        trans_err = float(np.linalg.norm(offset[:3, 3]) * 1000.0)
        theta_err = abs(np.degrees(theta - frame["theta_rad"]))
        theta_err = min(theta_err, 360.0 - theta_err)

        rows.append({"theta_gt_deg": frame["theta_deg"], "theta_err_deg": theta_err,
                     "body_rot_err_deg": rot_err, "body_trans_err_mm": trans_err,
                     "hinge_residual": residual})
        print(f"{frame['theta_deg']:9.1f} {theta_err:9.2f}d {rot_err:8.2f}d "
              f"{trans_err:10.2f}mm {residual:12.4f} {elapsed:6.1f}", flush=True)

    (args.synthetic / "probe_fpose_results.json").write_text(
        json.dumps({"camera": camera_id, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
