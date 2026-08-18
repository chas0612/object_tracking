#!/usr/bin/env python3
"""Stage 2: first articulated pose on real footage. FoundationPose proposes, silhouettes decide.

The two halves that worked on the synthetic fixture, now on a real capture:

* **FoundationPose** turns one stereo view's RGB-D into a 6D pose. On synthetic
  data, given the correct lid angle, it reached 0.4-0.8 degrees and 0.5-1.3 mm,
  and it needs no global orientation search -- which is exactly where silhouette
  fitting alone failed on this same real frame, converging 176 degrees off.
* **Multi-view silhouette IoU** picks the lid angle. FoundationPose's own score
  cannot: it ranks poses within a single mesh and is not commensurable across the
  different meshes that different angles produce. Measured on synthetic data,
  using it chose 150/120/0 degrees against truths of 0/90/180; swapping in the
  silhouette yardstick got all three exactly right.

Depth is read from disk, written earlier by ``real_depth.py``. It is not computed
here on purpose: TensorRT stereo needs ``pycuda.autoinit`` and its CUDA context
makes NVIDIA Warp resolve devices as ``cuda:0.0``, so ``erode_depth_kernel``
refuses to launch on arrays that live on ``cuda:0``.

The lid angle is chosen **once, for all pairs together**. Letting each pair take
its own argmax off the coarse sweep is what failed at frame 108 of ep2: at
512x384 with 4000 samples, three of four pairs preferred 150 deg over 165 by 0.001
IoU, which is well inside the noise of a point-splat score; at 1024x768 with 12000
the ranking inverts and 164 deg wins by 0.005. So the coarse sweep now only screens
-- the top few angles per pair are re-scored on the fine objective and one global
argmax over (pair, angle) decides. Pairs whose registered pose at that angle sits
far from the medoid are dropped, by medoid rather than by a maximum, because one
flipped registration out of four must not be able to veto an angle.

Each pair still registers and refines from its own depth view, so agreement between
the surviving answers remains a check across independent measurements. Only the
angle is shared, and it was never independent between pairs anyway: every pair
scored it against the same masks.

Depth is used to *propose* poses and never to rank them. Fitting pose and angle to
the fused stereo cloud looks like an independent arbiter and is not: on the closed
frame, where the answer is known, that fit moves the lid to -6.9 deg -- outside the
joint's range, physically impossible -- to buy 0.2 mm of residual. See
``FINDINGS.md`` in the capture directory.

The silhouette does not know the hinge exists, so the joint's range has to be
imposed on it from outside. Left free, the refinement took the lid to -15 deg at
frame 72 and won a tie-break on the IoU that bought -- the same failure that
discredited the depth objective, arrived at from the other direction. Any search
over the angle is bounded by the joint's [theta_min, theta_max] range.

``--frames`` solves a series in one process. Every frame is solved from scratch --
no pose, angle or seed carries over -- because the series exists to test the
method, not to smooth it. Agreement between pairs and between measures is
self-consistency, and this pipeline has already produced a self-consistent wrong
answer once. A lid angle recovered independently per frame has no reason to trace
a smooth curve unless it is tracking the real lid, so the shape of that curve is
evidence of a kind the single-frame checks cannot give. Sharing the estimators is
safe on those terms: they depend on the angle grid and the mesh, not on the frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

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

from common import (joint_grid, load_articulation, load_cameras,  # noqa: E402
                    theta_grid)
from depth_joint import DepthJointObjective  # noqa: E402
from fit_rc import SilhouetteObjective, _from_pose, _to_pose  # noqa: E402
from probe_fpose_theta import _fused  # noqa: E402
from real_first_pose import DEFAULT_CAPTURE, _load_masks, _overlay_sheet  # noqa: E402


def _read_frame(video: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {index} from {video}")
        return frame
    finally:
        capture.release()


def _prepare_parts(articulation, target_faces: int):
    """Decimate each part once, so fusing per angle costs nothing.

    ``probe_fpose_theta._fused`` welds the full 240k-face body to the 86k-face lid
    and decimates the 326k result, once per angle hypothesis. That is where the
    build time goes, and it is pure waste: decimation does not depend on the
    angle. Splitting the budget between the parts by face count gives the same
    total and reduces the work to two decimations for the whole sweep.
    """
    from probe_fpose import _decimate

    body_faces = len(articulation.body.faces)
    lid_faces = len(articulation.lid.faces)
    share = target_faces / (body_faces + lid_faces)
    return (_decimate(articulation.body, int(body_faces * share)),
            _decimate(articulation.lid, int(lid_faces * share)))


def _fuse_prepared(body, lid, articulation, theta: float):
    """Weld the pre-decimated parts with the lid swung to ``theta``."""
    import trimesh

    swung = lid.copy()
    swung.apply_transform(articulation.joint_transform(theta))
    return trimesh.util.concatenate([body, swung])


def _block_reduce_depth(depth: np.ndarray, factor: int) -> np.ndarray:
    """Shrink a depth map by taking the nearest valid sample in each block.

    Reprojected stereo depth is not patchy, it is *diluted*: the engine solves
    disparity at 672x448 and those ~301k points scatter into a 2048x1536 target,
    covering 7.8% of it. Grouping 4x4 recovers 97.4% coverage -- the information
    was always there, the grid was just four times too fine to hold it.

    This matters because ``FoundationPose.register`` erodes the depth by radius 2
    before using it. Every isolated sample has invalid neighbours, so erosion
    wipes the map and registration bails out with "valid too small".

    Nearest (minimum of the valid samples) rather than mean: at a silhouette edge
    a block straddles object and table, and the object is the surface we want.
    """
    if factor <= 1:
        return depth
    height = depth.shape[0] // factor * factor
    width = depth.shape[1] // factor * factor
    blocks = (depth[:height, :width]
              .reshape(height // factor, factor, width // factor, factor)
              .transpose(0, 2, 1, 3)
              .reshape(height // factor, width // factor, factor * factor))
    nearest = np.where(blocks > 0, blocks, np.inf).min(axis=-1)
    return np.where(np.isfinite(nearest), nearest, 0.0).astype(np.float32)


def _joint_bounds(articulation, args) -> tuple[float, float]:
    """The sweep's bounds, in the joint's own units, with the CLI override applied.

    Two pairs of flags, one per unit, and each refuses the joint it does not describe
    rather than being reinterpreted. An angle read as a length -- or the reverse -- is
    wrong by a factor of the object's size, produces a plausible-looking sweep, and
    raises nowhere. ``--joint-min``/``--joint-max`` are in the mesh's own units, the
    same ones ``joint.json`` states its range in and the tracker takes
    ``--init-joint-value`` in. Only the printing is in millimetres.
    """
    revolute = articulation.joint_type == "revolute"
    if revolute and (args.joint_min is not None or args.joint_max is not None):
        raise ValueError("--joint-min/--joint-max are lengths and this joint is "
                         "revolute; use --theta-min-deg/--theta-max-deg")
    if not revolute and (args.theta_min_deg is not None or args.theta_max_deg is not None):
        raise ValueError("--theta-min-deg/--theta-max-deg are angles and this joint is "
                         f"{articulation.joint_type}; use --joint-min/--joint-max, in "
                         "the same units as the mesh")
    if revolute:
        lower = (articulation.joint_min if args.theta_min_deg is None
                 else float(np.radians(args.theta_min_deg)))
        upper = (articulation.joint_max if args.theta_max_deg is None
                 else float(np.radians(args.theta_max_deg)))
    else:
        lower = articulation.joint_min if args.joint_min is None else float(args.joint_min)
        upper = articulation.joint_max if args.joint_max is None else float(args.joint_max)
    if lower > upper:
        raise ValueError(
            f"Invalid joint range {articulation.display(lower):.1f} .. "
            f"{articulation.display(upper):.1f} {articulation.joint_unit}")
    return lower, upper


def _joint_sweep(articulation, lower: float, upper: float, args) -> np.ndarray:
    """The coarse grid, in the joint's own units.

    A revolute sweep steps in degrees and keeps the arithmetic it always had. A
    prismatic one has no step it can inherit -- 15 degrees means nothing about a
    drawer -- so absent ``--joint-step`` it takes a twelfth of the joint's own range,
    which lands near the number of samples the hinge uses and cannot be wrong by a
    change of unit system.
    """
    if articulation.joint_type == "revolute":
        return theta_grid(lower, upper, args.theta_step_deg)
    step = (float(args.joint_step) if args.joint_step is not None
            else (upper - lower) / 12.0)
    if step <= 0:
        raise ValueError("--joint-step must be positive")
    return joint_grid(lower, upper, step)


def _scan_step(articulation, lower: float, upper: float, args) -> float:
    """Step of the tie-break scan, in the joint's own units. 0 disables it."""
    if articulation.joint_type == "revolute":
        return float(np.radians(args.theta_scan_deg))
    if args.joint_scan is not None:
        return float(args.joint_scan)
    # The hinge scans its whole range at 1 degree, a few hundred samples. Matching
    # the sample count rather than the number keeps the cost the same and the unit
    # out of it.
    return (upper - lower) / 250.0


# How far the silhouette may drag the body away from the pose depth gave it.
# Sized from FoundationPose's own measured accuracy on the synthetic fixture --
# 0.4-0.8 deg and 0.5-1.3 mm -- so this is about twice its error bar: enough for a
# real correction, not enough to fund a different answer. At three times this the
# refinement spent 2.2-2.4 deg of body rotation at frame 108 and carried the lid
# from 165 to 161 deg with it.
BODY_ROTATION_SLACK = 0.025     # radians per rotation-vector component, about 1.4 deg
BODY_TRANSLATION_SLACK = 0.004  # metres


def _polish(fine, pose: np.ndarray, theta: float,
            theta_min: float, theta_max: float):
    """Optimise the six pose parameters and the lid angle as one seven-vector.

    Genuinely together, which the earlier version of this only claimed. It fitted
    the pose at a fixed angle, swept the angle with that pose frozen, then fitted
    the pose again -- and a pose already fitted to an angle has absorbed the
    mismatch, so every other angle scores worse and the sweep hands back the value
    it was given. Both real frames showed the symptom: 45.0 deg in and 45.0 deg
    out at frame 72, 150 to 151 at frame 108. The lid turns about 2.5 deg per
    frame, so an angle pinned to the 15 deg search grid cannot track it, and a
    trajectory made of those would be a staircase that passes a smoothness test
    while measuring nothing.

    The angle is bounded by the joint's range rather than clipped inside the cost.
    Left free it walked to -15 deg at frame 72 and won a tie-break on the IoU that
    bought -- the same failure that discredited the depth objective, reached from
    the other side. Nothing in a silhouette knows the hinge exists, so the limit
    has to come from outside it.

    The body is bounded too, near the pose FoundationPose derived from depth, and
    that boundary is the point rather than a safeguard. This pipeline divides the
    work: depth proposes the pose, the silhouette decides the angle. Depth is
    metric and pins all six body parameters; an outline barely constrains distance
    along the viewing ray or any rotation that preserves it, so handing the body to
    the silhouette discards the measurement that fixed it. Left free at frame 108
    it spent that freedom -- moved the body 4.0 deg / 4.0 mm, gained 0.0044 of
    pooled IoU, and carried the lid from 165 to 160 deg. Freezing the body and
    sweeping the angle alone showed what the views actually wanted: 18 of 21 peak
    between 162 and 169 deg, median 166. The angle was never in doubt; the body
    drift dragged it.
    """
    start = np.concatenate([_from_pose(pose), [theta]])
    # Per-component limits on the rotation vector rather than on the rotation
    # angle: for deviations this small the two agree closely, and a box is what
    # the optimiser can take directly.
    bounds = ([(value - BODY_ROTATION_SLACK, value + BODY_ROTATION_SLACK)
               for value in start[:3]]
              + [(value - BODY_TRANSLATION_SLACK, value + BODY_TRANSLATION_SLACK)
                 for value in start[3:6]]
              + [(float(theta_min), float(theta_max))])
    # Powell, restarted until it stops finding anything. Its stopping rule is
    # relative to the function value, so at an IoU near 0.55 "converged" means an
    # improvement under about 5e-5, which one sweep of the direction set reaches on
    # a bumpy surface while there is still ground to make up. That is why one start
    # at frame 72 finished in 18 s below the score it began with while its twin took
    # 42 s and did better, and why the two then disagreed by 8.9 deg -- a
    # disagreement about where the optimiser stopped, not about where the lid was.
    # Restarting rebuilds the direction set from the current point, and keeping the
    # best seen means this can no longer return something worse than its input.
    vector = start
    value = fine.cost(start[:6], float(start[6]))
    for _ in range(4):
        result = minimize(lambda x: fine.cost(x[:6], float(x[6])), vector, method="Powell",
                          bounds=bounds, options={"maxiter": 65, "xtol": 1e-3, "ftol": 1e-4})
        if result.fun >= value - 1e-5:
            break
        vector, value = np.asarray(result.x), float(result.fun)
    return _to_pose(vector[:6]), float(vector[6]), -value


def _polish_body(fine, pose: np.ndarray, theta: float):
    """The six body parameters only, with the joint coordinate held where it is.

    The counterpart to :func:`_polish` for a joint whose coordinate the silhouette is
    not entitled to decide. Handing the seven-vector to Powell when the mask omits the
    moving part does not merely fail to improve the joint -- it actively walks it back,
    because hiding the unmasked part inside the body is what raises the outline score.
    Freezing it keeps the two objectives on the axes each can actually measure.

    The body's bounds are the same as in :func:`_polish` and for the same reason: the
    pose is a depth measurement and the silhouette only refines it.
    """
    start = _from_pose(pose)
    bounds = ([(value - BODY_ROTATION_SLACK, value + BODY_ROTATION_SLACK)
               for value in start[:3]]
              + [(value - BODY_TRANSLATION_SLACK, value + BODY_TRANSLATION_SLACK)
                 for value in start[3:6]])
    vector = start
    value = fine.cost(start, float(theta))
    for _ in range(4):
        result = minimize(lambda x: fine.cost(x, float(theta)), vector, method="Powell",
                          bounds=bounds, options={"maxiter": 65, "xtol": 1e-3, "ftol": 1e-4})
        if result.fun >= value - 1e-5:
            break
        vector, value = np.asarray(result.x), float(result.fun)
    return _to_pose(vector), float(theta), -value


def _pose_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(a) @ b
    angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    return angle, float(np.linalg.norm(delta[:3, 3]) * 1000.0)


def _inliers(poses: dict[str, np.ndarray], tolerance_deg: float) -> list[str]:
    """The pairs whose registered pose sits near the medoid of all of them.

    Medoid rather than mean or maximum: these are rigid poses, a flipped
    registration lands ~180 deg away, and one flip out of four must neither drag
    a centre nor veto the other three.
    """
    names = list(poses)
    if len(names) < 3:
        return names
    total = {name: sum(_pose_delta(poses[name], poses[other])[0]
                       for other in names if other != name)
             for name in names}
    medoid = min(total, key=total.get)
    return [name for name in names
            if _pose_delta(poses[medoid], poses[name])[0] <= tolerance_deg]


def _solve_frame(args, frame_index: int, articulation, full_cameras, estimators) -> dict | None:
    """One frame, start to finish. Nothing here carries over from the previous frame.

    That independence is the point when several frames are run together: the
    estimators and the mesh decimation are shared because they do not depend on
    the frame, but no pose, angle or seed is. A smooth angle trajectory out of
    this is then evidence, not something the pipeline arranged.
    """
    # The joint's upper limit comes from registering the closed scan against the
    # open one, so it is only as open as that scan happened to be. On this episode
    # the lid ends up flat and frames 132 onward pin to 205.8 deg exactly, which is
    # the model's edge rather than the hinge's -- those estimates are censored, not
    # measured. ``--theta-max-deg`` raises the ceiling, and it has to reach the
    # refinement and not only the coarse grid, or the sweep explores angles the
    # polish is still forbidden to keep.
    theta_min, theta_max = _joint_bounds(articulation, args)
    unit = articulation.joint_unit
    frame_dir = args.capture_dir / f"foundpose_frame_{frame_index:06d}"
    probe_dir = args.capture_dir / "articulated_probe" / f"frame_{frame_index:06d}"
    manifest_path = probe_dir / "depth" / "depth_manifest.json"
    if not manifest_path.exists():
        print(f"[skip] frame {frame_index}: no depth manifest at {manifest_path}", flush=True)
        return None
    depth_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = (args.output_root if args.output_root is not None
                   else args.capture_dir / "articulated_probe")
    out_dir = output_root / f"frame_{frame_index:06d}" / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    masks, score_cameras = _load_masks(frame_dir / "masks", full_cameras, args.scale)
    objective = SilhouetteObjective(articulation, score_cameras, masks, args.samples,
                                    backend=args.objective_backend)
    reference = next(iter(score_cameras.values()))
    print(f"\n{'=' * 78}\nframe {frame_index}: silhouette yardstick {len(masks)} views @ "
          f"{reference.width}x{reference.height}", flush=True)

    # A sliding joint gets its coordinate from depth instead of from the silhouette.
    # The reason is measured, not stylistic: on drawer/2 the mask held the cabinet
    # without the extended drawer, so the outline was best explained by a shut drawer
    # and the seed returned 0.6 mm against a truth near 220. Depth is computed over
    # the whole image and does not care what was segmented. See depth_joint.py.
    #
    # Revolute is deliberately left alone. Its masks have been complete on every
    # object run so far and its silhouette ranking is what those results rest on.
    depth_joint = None
    if articulation.joint_type == "prismatic" and not args.no_joint_depth:
        depth_joint = DepthJointObjective(
            articulation, manifest_path,
            json.loads((args.capture_dir / "cam_param/intrinsics.json").read_text(
                encoding="utf-8")),
            json.loads((args.capture_dir / "cam_param/extrinsics.json").read_text(
                encoding="utf-8")),
            scale=args.joint_depth_scale, tolerance_m=args.joint_depth_tolerance / 1000.0)
        print(f"frame {frame_index}: joint ranked on depth agreement over "
              f"{len(depth_joint.views)} pairs, tolerance "
              f"{args.joint_depth_tolerance:.0f} mm (mask not used)", flush=True)

    def rank(pose: np.ndarray, joint_value: float, iou: float) -> float:
        """What decides the joint coordinate. The silhouette, unless depth is on."""
        return iou if depth_joint is None else depth_joint.agreement(pose, joint_value)

    video_dir = args.capture_dir / "undistorted_video"
    factor = int(round(1.0 / args.fpose_scale))

    # Phase 1: every pair registers against every angle. This is the expensive
    # part and it is unchanged -- what changes is that nothing is decided here.
    sweeps, meta = {}, {}
    for pair, entry in depth_manifest["pairs"].items():
        camera_id = entry["reference_camera"]
        camera = full_cameras[camera_id].scaled(args.fpose_scale)
        depth = _block_reduce_depth(np.load(entry["depth_npy"]).astype(np.float32), factor)
        mask_full = cv2.imread(str(frame_dir / "masks" / f"{camera_id}.png"),
                               cv2.IMREAD_GRAYSCALE)
        if mask_full is None:
            # Stereo depth itself does not need a mask, so real_depth keeps pairs
            # whose SAM view failed.  FoundationPose registration does need one;
            # omit only that pair from the pose-proposal sweep while retaining its
            # depth in DepthJointObjective's global joint score.
            print(f"\npair {pair}  reference={camera_id}: no SAM mask; "
                  "skipping FoundationPose proposals for this pair", flush=True)
            continue
        mask = cv2.resize(mask_full, (camera.width, camera.height),
                          interpolation=cv2.INTER_AREA) > 127
        rgb = cv2.cvtColor(cv2.resize(
            _read_frame(video_dir / f"{camera_id}.avi", frame_index),
            (camera.width, camera.height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        depth = depth[:camera.height, :camera.width]
        mask = mask[:depth.shape[0], :depth.shape[1]]
        rgb = rgb[:depth.shape[0], :depth.shape[1]]
        filled = int(((depth > 0) & mask).sum())
        print(f"\npair {pair}  reference={camera_id}  {depth.shape[1]}x{depth.shape[0]}  "
              f"depth-in-mask {entry['valid_in_mask_px']} px at full res -> "
              f"{filled} px ({100 * filled / max(1, mask.sum()):.1f}% of mask)", flush=True)

        started = time.perf_counter()
        rows = []
        for theta, estimator in estimators:
            estimator.scores = None          # register() leaves it stale if it bails
            pose_camera = estimator.register(K=camera.K.astype(np.float64), rgb=rgb,
                                             depth=depth, ob_mask=mask,
                                             iteration=args.refine_iterations)
            if getattr(estimator, "scores", None) is None:
                # register() returned early -- not enough usable depth for this view.
                print(f"    theta={articulation.display(theta):6.1f}  registration bailed out",
                      flush=True)
                continue
            pose_world = np.linalg.inv(camera.extrinsic) @ np.asarray(pose_camera)
            iou = objective.iou(pose_world, theta)
            native = float(estimator.scores[0].detach().cpu())
            rows.append({"joint_disp": float(articulation.display(theta)), "silhouette_iou": iou,
                         "foundationpose_score": native, "pose_body": pose_world.tolist()})
            print(f"    theta={articulation.display(theta):6.1f}  IoU={iou:.4f}  fp_score={native:8.2f}",
                  flush=True)

        if not rows:
            print(f"  no theta registered for {pair}; skipping", flush=True)
            continue
        sweeps[pair] = rows
        best_row = max(rows, key=lambda row: row["silhouette_iou"])
        meta[pair] = {"reference_camera": camera_id,
                      "depth_valid_px": entry["valid_in_mask_px"],
                      "coarse_joint_disp": best_row["joint_disp"],
                      "coarse_iou_own_scale": best_row["silhouette_iou"],
                      "fp_score_joint_disp": max(
                          rows, key=lambda row: row["foundationpose_score"])["joint_disp"],
                      "sweep_seconds": time.perf_counter() - started}
        print(f"  swept in {time.perf_counter() - started:.0f}s, "
              f"coarse best {best_row['joint_disp']:.0f}{unit}", flush=True)

    if not sweeps:
        print(f"frame {frame_index}: no pair registered at any angle", flush=True)
        return None

    # Phase 2: one angle for all pairs, chosen at the refine resolution.
    #
    # Letting each pair take its own coarse argmax is what went wrong at frame
    # 108: at 512x384 with 4000 samples three pairs preferred 150 deg over 165 by
    # 0.001, and at 1024x768 with 12000 the ranking inverts and 164 deg wins
    # outright. The coarse score is a screen, not a decision. So keep the top few
    # angles per pair, re-score that shortlist on the fine objective, and take one
    # global argmax over (pair, angle) -- the same yardstick for every candidate.
    fine_masks, fine_cameras = _load_masks(frame_dir / "masks", full_cameras,
                                           args.refine_scale or args.scale)
    fine = SilhouetteObjective(articulation, fine_cameras, fine_masks, args.refine_samples,
                               backend=args.objective_backend)
    fine_camera = fine_cameras[next(iter(fine_cameras))]

    shortlist = sorted({row["joint_disp"] for rows in sweeps.values()
                        for row in sorted(rows, key=lambda r: -r["silhouette_iou"])[
                            :args.shortlist]})
    by_theta = {pair: {row["joint_disp"]: row for row in rows} for pair, rows in sweeps.items()}
    scored_by = "depth" if depth_joint is not None else "IoU"
    print(f"\nglobal angle selection at {fine_camera.width}x{fine_camera.height} "
          f"(ranked on {scored_by}), shortlist {[f'{t:.0f}' for t in shortlist]}", flush=True)
    print("   theta  " + " ".join(f"{p.split(':')[0][-4:]:>8}" for p in sweeps)
          + "   agree", flush=True)
    # ``grid`` stays the silhouette, because that is what phase 4 refines on and what
    # the record reports. ``ranked_grid`` is what chooses the joint coordinate, and
    # the two are the same object unless depth is on.
    grid, ranked_grid, consensus = {}, {}, {}
    for joint_disp in shortlist:
        cells = []
        for pair in sweeps:
            row = by_theta[pair].get(joint_disp)
            if row is None:
                cells.append("       -")
                continue
            pose = np.asarray(row["pose_body"])
            value = fine.iou(pose, articulation.from_display(joint_disp))
            grid[(pair, joint_disp)] = value
            ranked_grid[(pair, joint_disp)] = rank(
                pose, articulation.from_display(joint_disp), value)
            cells.append(f"{ranked_grid[(pair, joint_disp)]:8.4f}")
        consensus[joint_disp] = _inliers(
            {pair: np.asarray(by_theta[pair][joint_disp]["pose_body"])
             for pair in sweeps if joint_disp in by_theta[pair]}, args.reject_deg)
        print(f"  {joint_disp:6.1f}  " + " ".join(cells)
              + f"   {len(consensus[joint_disp])}/{len(sweeps)}", flush=True)

    # An angle is credible only if a majority of the pairs, each registering from
    # its own depth view, agree on *where the object is* at that angle.
    #
    # Scoring an angle by its single best pair is what broke frame 72. At 0 deg
    # one pair of four landed on a pose scoring 0.545 while the other three went
    # 83-110 deg elsewhere; at 45 deg all four agreed to within 0.014 IoU. A max
    # cannot tell those apart -- it sees 0.545 beating 0.546 and calls it a tie --
    # but one of them is a lucky degenerate fit and the other is a measurement
    # four independent views concur on. Agreement is the part that cannot be had
    # by accident, so it gates the angle rather than commenting on it afterwards.
    quorum = len(sweeps) // 2 + 1
    credible = [joint_disp for joint_disp in shortlist
                if len(consensus.get(joint_disp, [])) >= quorum]
    if not credible:
        print(f"  ** no angle reaches {quorum}/{len(sweeps)} pose agreement; "
              f"falling back to the unfiltered score", flush=True)
        credible = list(shortlist)
    elif len(credible) < len(shortlist):
        print(f"  angles reaching {quorum}/{len(sweeps)} agreement: "
              f"{[f'{t:.0f}' for t in credible]}", flush=True)

    ranked = {key: value for key, value in ranked_grid.items()
              if key[1] in credible and key[0] in consensus[key[1]]}
    best_pair, joint_star_disp = max(ranked, key=ranked.get)
    coarse_picks = {meta[p]["coarse_joint_disp"] for p in sweeps}
    print(f"  -> leading value {joint_star_disp:.1f} {unit} on {best_pair.split(':')[0]}, "
          f"{scored_by} {ranked[(best_pair, joint_star_disp)]:.4f} "
          f"(IoU {grid[(best_pair, joint_star_disp)]:.4f}); per-pair coarse would have "
          f"picked {sorted(coarse_picks)}", flush=True)

    # A thin lead is not a decision either. At frame 108 the leader is 0.0022 ahead
    # of the runner-up here, and 0.0022 is the same order as the gap that sent the
    # coarse stage to the wrong answer. Refinement separates them by three times
    # that, so when the shortlist is close, refine each contender on the leading
    # pair and let the finished fits decide.
    leaders = {}
    for (pair, joint_disp), value in ranked.items():
        leaders[joint_disp] = max(leaders.get(joint_disp, -np.inf), value)
    contenders = [joint_disp for joint_disp in sorted(leaders, key=leaders.get, reverse=True)
                  if leaders[joint_disp] >= leaders[joint_star_disp] - args.tie_margin
                  ][:args.max_contenders]
    tie_break = {}
    # With depth deciding the joint the scan is not a tie-break, it is the answer:
    # the shortlist is a 13-point grid over a 200 mm travel and the peak sits between
    # its points. So it runs whether or not the leaders are close.
    scan_always = depth_joint is not None
    if _scan_step(articulation, theta_min, theta_max, args) > 0 and (
            scan_always or len(contenders) > 1):
        # Every contender is scanned from the *same* pair's registration, and that
        # pair must be one that agrees with the others at every contending angle.
        # Taking each angle's own best pair instead is the other half of what broke
        # frame 72: 0 deg was refined from 22645021 and 45 deg from 25452066, so
        # the comparison was partly between two pairs' registration quality rather
        # than between the two angles.
        #
        # The scan replaces a bounded 7-DoF Powell polish per contender. Those cost
        # ~600 evaluations each and produced, between them, exactly one number: which
        # of the contending grid angles to keep. Sweeping theta with the body frozen
        # answers that from ~275 evaluations for the whole range and shows the shape
        # of the curve rather than three samples of it -- which matters here, because
        # the runners-up are not isolated peaks but a broad band on the other side of
        # the joint, and a polish that walks into one reports a number that hides it.
        # The body is not touched: it is what depth measured, and phase 4 is where it
        # is allowed to move.
        eligible = [p for p in sweeps
                    if all(p in consensus.get(joint_disp, []) for joint_disp in contenders)]
        reference_pair = max(eligible or list(sweeps),
                             key=lambda p: sum(grid.get((p, t), 0.0) for t in contenders))
        scan_step = _scan_step(articulation, theta_min, theta_max, args)
        scan = articulation.display(
            np.arange(theta_min, theta_max + 1.0e-12, scan_step))
        print(f"  contenders within {args.tie_margin} of the lead: "
              f"{sorted(f'{t:.0f}' for t in contenders)} -- scanning the joint from "
              f"{reference_pair.split(':')[0]} at "
              f"{articulation.display(scan_step):.3g} {unit} "
              f"({scan.size} values each) on {scored_by}", flush=True)
        for joint_disp in sorted(contenders):
            row = by_theta[reference_pair].get(joint_disp)
            if row is None:
                continue
            pose = np.asarray(row["pose_body"])
            values = np.array([
                rank(pose, articulation.from_display(t),
                     fine.iou(pose, articulation.from_display(t)) if depth_joint is None
                     else 0.0)
                for t in scan])
            top = int(np.argmax(values))
            tie_break[joint_disp] = {"landed_disp": float(scan[top]), "iou": float(values[top]),
                                    "pair": reference_pair, "scored_by": scored_by,
                                    "scan_step": articulation.display(scan_step),
                                    "iou_at_start": float(
                                        values[int(np.argmin(np.abs(scan - joint_disp)))])}
            print(f"    from {joint_disp:6.1f} {unit} -> {scan[top]:6.1f} {unit}, "
                  f"scanned {scored_by} {values[top]:.4f}", flush=True)
        if tie_break:
            joint_star_disp = max(tie_break, key=lambda t: tie_break[t]["iou"])
            print(f"  -> joint {joint_star_disp:.1f} {unit} after the scan "
                  f"(peak at {tie_break[joint_star_disp]['landed_disp']:.1f} {unit})", flush=True)
    # The grid angle stays the key -- the registered poses are stored under it --
    # but phase 4 starts from where the scan actually peaked. Handing it the grid
    # point instead throws away the one thing the scan measured: on red_bowl/1 the
    # peak sat 3.8 deg off the grid and 0.0018 IoU above it, and phase 4's Powell
    # stopped without recovering the difference. The optimiser's stopping rule is
    # relative to the function value, so it cannot be relied on to walk back to a
    # maximum that is already known.
    theta_star = articulation.from_display(
        joint_star_disp if not tie_break else tie_break[joint_star_disp]["landed_disp"])

    # The values phase 4 re-scans the joint over, in the joint's own units. Same grid
    # the tie-break used, so the refinement cannot land somewhere the scan never
    # looked; unused when the silhouette owns the joint.
    sweep_step = _scan_step(articulation, theta_min, theta_max, args)
    sweep_values = (np.arange(theta_min, theta_max + 1.0e-12, sweep_step)
                    if sweep_step > 0 else np.array([theta_star]))

    # Phase 3: keep the pairs that agree at the chosen angle. This is the same
    # agreement already measured for every shortlisted angle above, so it is only
    # read back here -- at frame 108 it is what discards the one pair of four that
    # flipped 178 deg while the other three agreed to 12.
    available = [pair for pair in sweeps if joint_star_disp in by_theta[pair]]
    poses = {pair: np.asarray(by_theta[pair][joint_star_disp]["pose_body"])
             for pair in available}
    keep = [pair for pair in available if pair in consensus.get(joint_star_disp, available)]
    rejected = [pair for pair in available if pair not in keep]
    agreeing = len(keep)
    if rejected:
        centre = poses[keep[0]] if keep else poses[available[0]]
        for pair in rejected:
            print(f"  rejected {pair}: {_pose_delta(centre, poses[pair])[0]:.1f} deg "
                  f"from the agreeing pairs", flush=True)
    # The quorum above should make this unreachable, but it is cheap insurance and
    # it is the honest thing to print: an answer resting on a minority of the pairs
    # is a suspect angle, not a confident pose.
    if len(rejected) > agreeing:
        print(f"  ** {len(rejected)} of {len(available)} pairs disagree at "
              f"{joint_star_disp:.1f} {unit} -- suspect the angle, not the pairs",
              flush=True)
    # Each polish costs about as much as the sweep that fed it, so on a frame
    # series the cheapest honest thing is to start from the best-scoring survivors
    # rather than all of them.
    if args.refine_pairs:
        keep = sorted(keep, key=lambda pair: -grid.get((pair, joint_star_disp), 0.0)
                      )[:args.refine_pairs]

    # Phase 4: one answer, from as many starts as there are agreeing pairs.
    #
    # The lid has one angle, so a pipeline that returns one per stereo pair has not
    # finished. The pairs were never independent answers anyway -- they share the
    # masks and, since phase 2, the angle -- and treating them as votes invited
    # exactly the error this frame series found: at frame 72 two pairs came back
    # 8.9 deg apart, and the gap was the optimiser stopping in different places
    # (19 s against 42 s) rather than any disagreement about the lid.
    #
    # So the pairs become starting points for one optimisation and the best-scoring
    # finish is the answer. That is the standard remedy for a bumpy objective, it
    # costs what the per-pair version cost, and it turns the spread between starts
    # from a contradiction in the output into a diagnostic about the search.
    candidates = {}
    for pair in keep:
        started = time.perf_counter()
        pose, theta = poses[pair], theta_star
        coarse_pose = pose.copy()
        # Re-score the starting pose on *this* objective. IoU here is a point-splat
        # approximation whose absolute value falls as resolution rises even for an
        # unchanged pose, so comparing against the coarse number would read as a
        # regression that did not happen.
        baseline_iou = fine.iou(pose, theta)
        iou = baseline_iou
        if args.refine_scale > 0:
            if depth_joint is None:
                pose, theta, iou = _polish(fine, pose, theta, theta_min, theta_max)
            else:
                # Alternate rather than optimise jointly, because the two axes are
                # answered by two different measurements. Twice is enough: the body
                # moves by a millimetre or two here, and the joint's optimum is not
                # sensitive to that at the scale a slide covers.
                for _ in range(2):
                    pose, theta, iou = _polish_body(fine, pose, theta)
                    values = np.array([depth_joint.agreement(pose, t) for t in sweep_values])
                    theta = float(sweep_values[int(np.argmax(values))])
                iou = fine.iou(pose, theta)
            moved_rot, moved_mm = _pose_delta(coarse_pose, pose)
            print(f"\n  {pair}: joint {articulation.display(theta_star):.1f} -> "
                  f"{articulation.display(theta):.1f} {unit} (grid {joint_star_disp:.0f}), "
                  f"pose moved {moved_rot:.2f} deg / {moved_mm:.1f} mm, "
                  f"IoU {baseline_iou:.4f} -> {iou:.4f} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

        candidates[pair] = dict(meta[pair], pose_body=pose.tolist(),
                                joint_disp=float(articulation.display(theta)), silhouette_iou=iou,
                                iou_before_refine_same_scale=baseline_iou,
                                selected_joint_disp=joint_star_disp, sweep=sweeps[pair],
                                # Recorded whenever depth decided the joint, because
                                # then the reported IoU is no longer the number the
                                # answer was chosen by and reading it as one would
                                # mislead: on drawer/2 the right answer scores *worse*
                                # on IoU than the wrong one.
                                **({"joint_depth_agreement":
                                    float(depth_joint.agreement(pose, theta)),
                                    "joint_depth_pixels":
                                    int(depth_joint.visible_pixels(pose, theta))}
                                   if depth_joint is not None else {}))

    if not candidates:
        print(f"frame {frame_index}: no pair survived to refinement", flush=True)
        return None

    # The answer: the start that finished highest. Where the starts landed apart,
    # that is the search failing to converge, not the lid being in two places, and
    # the spread is reported as such.
    seed_pair = max(candidates, key=lambda pair: candidates[pair]["silhouette_iou"])
    answer = candidates[seed_pair]
    if len(candidates) > 1:
        angles = [c["joint_disp"] for c in candidates.values()]
        starts = sorted(candidates, key=lambda p: -candidates[p]["silhouette_iou"])
        print(f"\n  {len(candidates)} starts -> "
              + ", ".join(f"{candidates[p]['joint_disp']:.1f} {unit} @ "
                          f"{candidates[p]['silhouette_iou']:.4f}" for p in starts), flush=True)
        print(f"  answer: joint {answer['joint_disp']:.1f} {unit} from "
              f"{seed_pair.split(':')[0]}, IoU {answer['silhouette_iou']:.4f}; "
              f"starts spread {max(angles) - min(angles):.1f} {unit}", flush=True)
    else:
        print(f"\n  answer: joint {answer['joint_disp']:.1f} {unit}, "
              f"IoU {answer['silhouette_iou']:.4f}", flush=True)

    _overlay_sheet(out_dir / "overlay.png", frame_dir / "images", masks, score_cameras,
                   articulation, np.asarray(answer["pose_body"]),
                   articulation.from_display(answer["joint_disp"]))

    # `joint_disp` throughout this record is the joint coordinate in the unit named
    # by `joint_unit` -- degrees for a hinge, millimetres for a slide. `joint_value`
    # is the same quantity in the joint's own units, which is what the tracker takes,
    # and it is the one to read. `theta_deg` is kept only where it is true.
    record = {"capture_dir": str(args.capture_dir), "frame_index": frame_index,
              "object": args.object,
              "joint_type": articulation.joint_type,
              "joint_unit": articulation.joint_unit,
              "joint_value": articulation.from_display(answer["joint_disp"]),
              **({"theta_deg": answer["joint_disp"]}
                 if articulation.joint_type == "revolute" else {}),
              "selected_joint_disp": joint_star_disp, "tie_break": tie_break,
              "rejected_pairs": rejected, "pairs_available": len(available),
              "pairs_agreeing": agreeing, "angle_disputed": len(rejected) > agreeing,
              # Every pair's sweep, including the rejected ones. A disputed frame is
              # exactly the one worth re-reading, and keeping only the survivors
              # throws away the evidence that it was disputed.
              "consensus": {f"{t:.1f}": consensus[t] for t in sorted(consensus)},
              "sweeps": sweeps, "answer": answer, "answer_from": seed_pair,
              "starts": candidates,
              "start_spread_disp": (max(c["joint_disp"] for c in candidates.values())
                                   - min(c["joint_disp"] for c in candidates.values())),
              # Kept under the old key so the existing diagnostics keep working.
              "results": candidates}
    (out_dir / "hybrid_result.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")

    # Raycast the final answers onto the photographs. The point splats above show
    # where the mesh landed but not how well it fits: scattered points carry no
    # occlusion, so an overshooting silhouette looks like a correct one. Rendered
    # outlines make that judgeable by eye, which is the only check available on
    # footage with no ground truth.
    if args.render_scale > 0:
        # Imported here, not at module scope: this pulls in Open3D, and the fitting
        # above already has Warp, nvdiffrast and Torch holding CUDA state.
        from render_overlay import mesh_overlay_sheet, pick_views, shared_window

        render_cameras = {cid: cam.scaled(args.render_scale)
                          for cid, cam in full_cameras.items()}
        views = pick_views(render_cameras, 5)
        # One crop box across the answer and the starts it beat, so the rows stay
        # comparable -- cropping each to its own mesh would re-centre a drifted
        # start and hide the drift the sheet exists to show.
        windows = {cid: shared_window(render_cameras[cid], articulation,
                                      [(np.asarray(c["pose_body"]),
                                        articulation.from_display(c["joint_disp"]))
                                       for c in candidates.values()])
                   for cid in views}
        mesh_overlay_sheet(
            out_dir / "final.png", frame_dir / "images", render_cameras, articulation,
            np.asarray(answer["pose_body"]), articulation.from_display(answer["joint_disp"]),
            f"f{frame_index}  theta={answer['joint_disp']:.1f}deg  "
            f"IoU={answer['silhouette_iou']:.4f}", views, windows=windows)
        # The losing starts, only when they landed somewhere else: a start that
        # finished 9 deg away is the thing worth looking at, and when they all agree
        # there is nothing to see.
        if record["start_spread_disp"] > 1.0:
            sheets = []
            for pair in sorted(candidates, key=lambda p: -candidates[p]["silhouette_iou"]):
                start = candidates[pair]
                path = out_dir / f"start_{start['reference_camera']}.png"
                mesh_overlay_sheet(
                    path, frame_dir / "images", render_cameras, articulation,
                    np.asarray(start["pose_body"]), articulation.from_display(start["joint_disp"]),
                    f"start {pair.split(':')[0]}  theta={start['joint_disp']:.1f}deg  "
                    f"IoU={start['silhouette_iou']:.4f}", views, windows=windows)
                sheets.append(cv2.imread(str(path)))
            cv2.imwrite(str(out_dir / "starts.png"), np.vstack(sheets))
        print(f"rendered final mesh overlay over {len(views)} views", flush=True)

    print(f"\nwrote {out_dir}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--frame-index", type=int, default=40)
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="Solve several frames in one process, independently. Only the "
                             "estimators and the decimated parts are shared, and neither "
                             "depends on the frame; building them is most of a single run's "
                             "fixed cost.")
    parser.add_argument("--object", default="blue_plastic_box")
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help=("Root for frame_<index>/hybrid outputs. Depth is still read from "
              "<capture-dir>/articulated_probe, so a new attempt can reuse stereo "
              "depth without overwriting an earlier FoundationPose result."),
    )
    parser.add_argument("--theta-step-deg", type=float, default=15.0)
    parser.add_argument(
        "--joint-step", type=float, default=None,
        help="Prismatic joints only. Coarse sweep step, in the mesh's own units -- "
             "the ones joint.json states its range in. Defaults to a twelfth of that "
             "range. Separate from --theta-step-deg because 15 degrees says nothing "
             "about how far a drawer should slide between hypotheses.")
    parser.add_argument(
        "--joint-min", type=float, default=None,
        help="Prismatic joints only. Lower sweep bound in the mesh's own units; "
             "defaults to joint.json. The angular counterpart is --theta-min-deg and "
             "passing it for a prismatic joint is refused, not converted.")
    parser.add_argument(
        "--joint-max", type=float, default=None,
        help="Prismatic joints only. Upper sweep bound in the mesh's own units. Treat "
             "a published limit as a guard, not a measurement -- see --theta-max-deg.")
    parser.add_argument(
        "--joint-scan", type=float, default=None,
        help="Prismatic joints only. Step of the tie-break scan, in the mesh's own "
             "units. Defaults to a 250th of the range, which is the sample count the "
             "hinge's 1-degree scan uses over its own.")
    parser.add_argument("--theta-min-deg", type=float, default=None,
                        help="Override the sweep's lower bound. Defaults to the joint file; "
                             "negative values are valid when zero lies between two stops.")
    parser.add_argument("--theta-max-deg", type=float, default=None,
                        help="Override the sweep's upper bound. The joint was measured at "
                             "205.8 deg from the two-state scans, but the first real run "
                             "picked 195 deg -- the last value on the grid -- from all "
                             "three stereo pairs, which is what saturation looks like.")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Resolution for the silhouette yardstick.")
    parser.add_argument("--fpose-scale", type=float, default=0.25,
                        help="Resolution for FoundationPose. Matched to the reprojected "
                             "depth's real sample density (672x448 disparity scattered "
                             "into a 2048x1536 target), not chosen for speed.")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--decimate-faces", type=int, default=20000)
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--refine-scale", type=float, default=0.5,
                        help="Resolution for the final 7-DoF polish. 0 disables it.")
    parser.add_argument("--refine-samples", type=int, default=12000)
    parser.add_argument("--budget-gib", type=float, default=10.0)
    parser.add_argument("--render-scale", type=float, default=0.5,
                        help="Resolution for the final raycast mesh overlays. 0 skips them.")
    parser.add_argument("--shortlist", type=int, default=5,
                        help="Angles kept per pair from the coarse sweep and re-scored "
                             "at the refine resolution, where the ranking differs.")
    parser.add_argument("--tie-margin", type=float, default=0.02,
                        help="Angles scoring within this of the leader are scanned and "
                             "compared, rather than decided on the unrefined score.")
    parser.add_argument("--theta-scan-deg", type=float, default=1.0,
                        help="Step of the theta scan that breaks a tie between "
                             "contending angles, with the body frozen at what depth "
                             "measured. 0 disables the tie-break entirely, leaving the "
                             "fine grid's argmax to decide.")
    parser.add_argument(
        "--no-joint-depth", action="store_true",
        help="Rank a prismatic joint's coordinate on the silhouette, as a revolute one "
             "is ranked, instead of on measured depth. This is the pre-2026-08-13 "
             "behaviour and it is kept only to reproduce those runs: on drawer/2 it "
             "returned 0.6 mm against a truth near 220 because the mask held the "
             "cabinet without the extended drawer. Has no effect on a revolute joint, "
             "which never takes the depth path.")
    parser.add_argument(
        "--joint-depth-tolerance", type=float, default=20.0,
        help="Millimetres by which a predicted surface may differ from the measured "
             "depth and still count as agreeing. Wide because it absorbs the body "
             "pose's own error as well as the stereo's; the discrimination this "
             "provides comes from the moving part being displaced by ten times this, "
             "not from tight matching.")
    parser.add_argument(
        "--joint-depth-scale", type=float, default=0.25,
        help="Resolution the depth agreement is evaluated at, as a fraction of the "
             "captured frame. The score is a pixel fraction, so it is insensitive to "
             "this; the cost is not.")
    parser.add_argument("--objective-backend", choices=["auto", "numpy", "torch"],
                        default="auto",
                        help="How the silhouette objective counts covered pixels. "
                             "'torch' uses a GPU bitmap and is ~100x faster; 'numpy' is "
                             "bit-identical to the sorting version it replaced. 'auto' "
                             "takes torch when a CUDA device is visible. Force 'numpy' "
                             "to compare IoU values against a pre-2026-08 run: the two "
                             "agree to ~2e-5, which is far below any margin used here "
                             "but still moves the fourth decimal of a reported score.")
    parser.add_argument("--max-contenders", type=int, default=3)
    parser.add_argument("--reject-deg", type=float, default=25.0,
                        help="Drop a pair whose pose at the chosen angle is further than "
                             "this from the medoid. A flipped registration lands ~180 deg "
                             "out, so anything between the pairs' honest spread and that "
                             "works; this is not a tuned threshold.")
    parser.add_argument("--refine-pairs", type=int, default=0,
                        help="Polish only this many surviving pairs, nearest the medoid "
                             "first. 0 polishes all of them, which is the cross-check; a "
                             "small number is for frame series, where the polish costs more "
                             "than everything else combined.")
    parser.add_argument("--summary", type=Path, default=None,
                        help="Write a one-line-per-frame angle summary here.")
    args = parser.parse_args()

    frames = args.frames if args.frames else [args.frame_index]

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

    articulation = load_articulation(args.object)
    full_cameras = load_cameras(args.capture_dir)

    joint_lo, joint_hi = _joint_bounds(articulation, args)
    thetas = _joint_sweep(articulation, joint_lo, joint_hi, args)
    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
    part_body, part_lid = _prepare_parts(articulation, args.decimate_faces)
    parent_name, child_name = articulation.part_names
    print(f"parts decimated once: {parent_name} {len(part_body.faces)}, "
          f"{child_name} {len(part_lid.faces)} faces", flush=True)
    estimators = []
    for theta in thetas:
        mesh = _fuse_prepared(part_body, part_lid, articulation, float(theta))
        estimators.append((float(theta), FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=scorer, refiner=refiner, glctx=glctx, debug=0,
            debug_dir="/tmp/fpose_real")))
    print(f"built {len(estimators)} estimators, {articulation.joint_type} "
          f"{articulation.display(thetas).round(1)} {articulation.joint_unit}", flush=True)

    summary = []
    for frame_index in frames:
        started = time.perf_counter()
        record = _solve_frame(args, frame_index, articulation, full_cameras, estimators)
        if record is None:
            summary.append({"frame_index": frame_index, "joint_disp": None})
            continue
        summary.append({
            "frame_index": frame_index,
            # The answer, not an average of the starts. Averaging was a fusion that
            # lived only in this report while the pipeline still returned one pose
            # per pair -- it made a spread between starts look like a tighter number
            # than anything actually computed.
            "joint_disp": record["answer"]["joint_disp"],
            "start_spread": record["start_spread_disp"],
            "selected_joint_disp": record["selected_joint_disp"],
            "coarse_joint_disp": sorted({r["coarse_joint_disp"]
                                        for r in record["starts"].values()}),
            "silhouette_iou": record["answer"]["silhouette_iou"],
            "pairs_kept": len(record["starts"]),
            "pairs_rejected": record["rejected_pairs"],
            "angle_disputed": record["angle_disputed"],
            "seconds": time.perf_counter() - started,
        })

    if len(frames) > 1:
        print(f"\n{'=' * 78}\nangle trajectory\n"
              f"  frame   coarse grid        chosen   refined    IoU  pairs   time", flush=True)
        for row in summary:
            if row["joint_disp"] is None:
                print(f"  {row['frame_index']:5d}   (no result)", flush=True)
                continue
            grid = ",".join(f"{t:.0f}" for t in row["coarse_joint_disp"])
            flag = "  <-- disputed" if row["angle_disputed"] else ""
            print(f"  {row['frame_index']:5d}   {grid:<16s} {row['selected_joint_disp']:6.1f}   "
                  f"{row['joint_disp']:6.1f}  {row['silhouette_iou']:.4f}  "
                  f"{row['pairs_kept']:5d}  {row['seconds']:5.0f}s{flag}", flush=True)

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(
            {"capture_dir": str(args.capture_dir), "object": args.object,
             "joint_type": articulation.joint_type,
             "joint_unit": articulation.joint_unit,
             "joint_min": articulation.display(joint_lo),
             "joint_max": articulation.display(joint_hi),
             "frames": summary}, indent=2) + "\n",
            encoding="utf-8")
        print(f"wrote {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
