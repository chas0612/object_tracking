#!/usr/bin/env python3
"""FoundationPose for articulated init: let its own scorer resolve the lid angle.

The earlier ``probe_fpose.py`` ran FoundationPose once per part and recovered the
angle from the two rigid results. That needs a **per-part** mask, and on real
footage there is no way to get one: SAM3 was asked directly and could not split
this object -- prompting "box lid" returned 152k px against the whole object's
170k on the one view of three that produced anything at all, and "open container
body" returned nothing on any view.

So invert it. Fuse body and lid into a single rigid mesh at a hypothesised angle
and register that against the **object** mask we do have. One sweep per frame, no
part segmentation anywhere.

Selecting theta by ``estimator.scores`` does *not* work, and the reason matters.
That score ranks pose hypotheses **for one mesh**; it is never trained to compare
across meshes, so its scale is not commensurable between them. A wrong-theta mesh
can score well by fitting the visible surface and letting the rest hang in space
that no camera observes. Measured: picking theta this way chose 150/120/0 degrees
against truths of 0/90/180, with rotation error pinned near 179.5 degrees --
because the best fit of a wrong-shaped mesh to the observed depth is often the
flipped one.

Given the *correct* theta, though, FoundationPose is excellent here: 0.4-0.8
degrees and 0.5-1.3 mm, from a single depth view (``diag_flip.py``). So the fix is
to keep FoundationPose for pose and choose theta by a mesh-independent measure --
multi-view silhouette IoU, which applies the same yardstick to every candidate.
The two methods cover each other's gap: FoundationPose removes the global search
that silhouette fitting fails at, and the silhouettes decide the angle that
FoundationPose cannot.

Running this on the synthetic fixture first is deliberate: it has ground truth,
so we learn whether the idea works before investing in the stereo depth pipeline
that real footage would need.
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

from common import BODY, LID, Articulation, load_articulation, load_cameras  # noqa: E402
from fit_rc import SilhouetteObjective  # noqa: E402
from probe_fpose import _decimate  # noqa: E402


def _fused(articulation: Articulation, theta: float, target_faces: int) -> trimesh.Trimesh:
    """Body and lid welded into one rigid mesh with the lid swung to ``theta``."""
    lid = articulation.lid.copy()
    lid.apply_transform(articulation.joint_transform(theta))
    fused = trimesh.util.concatenate([articulation.body, lid])
    return _decimate(fused, target_faces) if target_faces else fused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=Path(__file__).parent / "synthetic_05")
    parser.add_argument("--budget-gib", type=float, default=10.0)
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--decimate-faces", type=int, default=20000)
    parser.add_argument("--theta-step-deg", type=float, default=15.0)
    parser.add_argument("--samples", type=int, default=4000,
                        help="Surface points per part for the silhouette yardstick.")
    args = parser.parse_args()

    import torch
    total = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.budget_gib / total))

    sys.path.insert(0, str(FP_ROOT))
    sys.path.insert(0, str(FP_ROOT / "mycpp/build"))
    import nvdiffrast.torch as dr
    import estimater as estimater_module
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

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

    thetas = np.radians(np.arange(0.0, np.degrees(articulation.theta_max) + 1e-6,
                                  args.theta_step_deg))
    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()

    print(f"camera={camera_id} {camera.width}x{camera.height}  "
          f"theta hypotheses={len(thetas)}  budget={args.budget_gib} GiB", flush=True)
    estimators = []
    for theta in thetas:
        mesh = _fused(articulation, float(theta), args.decimate_faces)
        estimators.append((float(theta), FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=scorer, refiner=refiner, glctx=glctx,
            debug=0, debug_dir="/tmp/fpose_probe")))
    print(f"built {len(estimators)} estimators", flush=True)

    rows = []
    for frame in truth["frames"]:
        frame_dir = args.synthetic / f"frame_{frame['frame']:03d}"
        parts = np.load(frame_dir / f"{camera_id}_parts.npy")
        rgb = np.load(frame_dir / f"{camera_id}_rgb.npy")
        depth = np.load(frame_dir / f"{camera_id}_depth.npy").astype(np.float32)
        mask = parts >= 0          # object silhouette only -- what SAM3 actually gives

        # Object masks in every view. This is the yardstick that is comparable
        # across theta hypotheses, which FoundationPose's own score is not.
        all_masks = {cid: np.load(frame_dir / f"{cid}_parts.npy") >= 0
                     for cid in truth["camera_ids"]}
        objective = SilhouetteObjective(articulation, cameras, all_masks, args.samples)

        started = time.perf_counter()
        best = (-np.inf, None, 0.0, -np.inf)
        for theta, estimator in estimators:
            pose_camera = estimator.register(K=camera.K.astype(np.float64), rgb=rgb,
                                             depth=depth, ob_mask=mask,
                                             iteration=args.refine_iterations)
            pose_world = np.linalg.inv(camera.extrinsic) @ np.asarray(pose_camera)
            iou = objective.iou(pose_world, theta)
            # ``scores`` is a CUDA tensor, sorted descending by register(). Kept only
            # to record how badly it would have chosen.
            native = float(estimator.scores[0].detach().cpu())
            if iou > best[0]:
                best = (iou, pose_world, theta, native)
        elapsed = time.perf_counter() - started

        score, pose, theta, native = best
        offset = np.linalg.inv(pose_true) @ pose
        rot_err = np.degrees(np.arccos(
            np.clip((np.trace(offset[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        trans_err = float(np.linalg.norm(offset[:3, 3]) * 1000.0)
        theta_err = abs(np.degrees(theta) - frame["theta_deg"])

        rows.append({"theta_gt_deg": frame["theta_deg"], "theta_err_deg": theta_err,
                     "rot_err_deg": rot_err, "trans_err_mm": trans_err,
                     "silhouette_iou": score, "foundationpose_score": native})
        print(f"  theta_gt={frame['theta_deg']:6.1f}  picked={np.degrees(theta):6.1f}  "
              f"theta_err={theta_err:5.1f}d  rot_err={rot_err:6.2f}d  "
              f"trans_err={trans_err:7.2f}mm  IoU={score:.4f}  ({elapsed:.0f}s)", flush=True)

    (args.synthetic / "probe_fpose_theta_results.json").write_text(
        json.dumps({"camera": camera_id, "theta_step_deg": args.theta_step_deg,
                    "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
