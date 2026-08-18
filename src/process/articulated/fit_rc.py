#!/usr/bin/env python3
"""Recover 7-DoF articulated pose from multi-view silhouettes alone.

Given the object's binary mask in each of the 22 calibrated views, solve for the
body pose ``T_body`` (6 DoF) and the lid angle ``theta`` (1 DoF) by maximising
silhouette agreement. No depth, no per-object onboarding, no learned weights --
only the meshes, the calibration, and masks of the kind SAM3 already produces.

Deliberately uses the **object** silhouette (body union lid), not per-part masks.
Per-part masks would make the problem far easier and we will not have them at
run time.

The search mirrors the structure proposed for the tracker's fit stage: theta is a
single dimension, so it is swept exhaustively rather than descended into, which
removes the local-minimum risk that makes 3-DoF rotation search expensive. For
each theta on a coarse grid the 6-DoF body pose is refined, the best theta is
kept, and theta is then polished on a fine grid.

``--seed-noise`` perturbs the starting pose away from ground truth to measure the
convergence basin: how coarse an initialiser this could be bolted onto.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from common import Articulation, Camera, load_articulation, load_cameras, theta_grid


def _subsample(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    """A fixed random subset of surface points, area-weighted."""
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=int(rng.integers(1 << 30)))
    return np.asarray(points, dtype=np.float64)


class SilhouetteObjective:
    """Negative mean silhouette IoU over all views, as a function of (pose, theta).

    Each view needs three counts: how many distinct pixels the candidate covers,
    how many of those lie inside the mask, and the mask's own area. Getting the
    first one is the entire cost of this objective. The candidate is a point set
    whose 2x2 splats overlap heavily -- 96k splats collapse to 28k distinct pixels
    at 1024x768 -- and the obvious way to count distinct entries is to sort them.
    Measured on the red_bowl seed, that sort was 71% of one evaluation and roughly
    60% of the whole first-pose stage.

    Both backends here avoid the sort and are selected by ``backend``:

    ``numpy`` writes each hit's ordinal into a scratch buffer and keeps the hit
    whose ordinal survived the collision, which is O(n) and gives **bit-identical**
    scores to the sorting version it replaces.

    ``torch`` scatters into a boolean bitmap on the GPU, where duplicates fold by
    construction and nothing has to be counted twice. It is about 100x faster and
    agrees with ``numpy`` to ~2e-5 -- fp32 rounding moving a handful of points
    across a pixel boundary, three orders of magnitude below the smallest margin
    any caller compares on. Absolute IoU values still shift in the fourth decimal,
    so a run to be compared numerically against an older one should force
    ``numpy``.
    """

    def __init__(
        self,
        articulation: Articulation,
        cameras: dict[str, Camera],
        masks: dict[str, np.ndarray],
        samples: int = 2500,
        seed: int = 0,
        backend: str = "auto",
    ) -> None:
        rng = np.random.default_rng(seed)
        self.articulation = articulation
        self.cameras = cameras
        self.body_pts = _subsample(articulation.body, samples, rng)
        self.lid_pts = _subsample(articulation.lid, samples, rng)
        self.evaluations = 0

        # Flattened masks. Scoring touches only the pixels a candidate projects
        # onto, so each evaluation costs O(points) rather than O(image area) --
        # the difference between ~100 ms and a few ms per evaluation.
        self.flat_masks = {cid: m.reshape(-1) for cid, m in masks.items()}
        self.mask_area = {cid: int(m.sum()) for cid, m in masks.items()}
        self.backend = self._setup_backend(backend)

    def _setup_backend(self, backend: str) -> str:
        wanted = str(backend).lower()
        if wanted not in {"auto", "numpy", "torch"}:
            raise ValueError(f"backend must be auto, numpy or torch, got {backend!r}")
        self._scratch = {cid: np.full(camera.width * camera.height, -1, dtype=np.int64)
                         for cid, camera in self.cameras.items()}
        if wanted == "numpy":
            return "numpy"
        unavailable = self._setup_torch()
        if unavailable is None:
            return "torch"
        if wanted == "torch":
            raise RuntimeError(f"torch backend unavailable: {unavailable}")
        return "numpy"

    def _setup_torch(self) -> str | None:
        """Prepare the GPU bitmap, or say why it cannot be used."""
        try:
            import torch
        except ImportError as exc:                                  # pragma: no cover
            return f"torch is not importable ({exc})"
        if not torch.cuda.is_available():
            return "no CUDA device is visible"
        order = list(self.cameras)
        if not order:
            return "there are no views"
        first = self.cameras[order[0]]
        if any(self.cameras[cid].width != first.width
               or self.cameras[cid].height != first.height for cid in order):
            return "the views do not share a resolution"

        device = torch.device("cuda")
        self._torch, self._order = torch, order
        self._width, self._height = first.width, first.height
        # One slot past the image, where points that miss it are parked. Scattering
        # them somewhere real would mark a pixel; dropping them per-view would need
        # a variable-length index, which is what this backend exists to avoid.
        self._park = self._width * self._height
        self._park_t = torch.tensor(self._park, dtype=torch.long, device=device)
        self._zero_t = torch.tensor(0, dtype=torch.long, device=device)
        self._projection = torch.as_tensor(
            np.stack([self.cameras[cid].K @ self.cameras[cid].extrinsic[:3, :]
                      for cid in order]), dtype=torch.float32, device=device)
        masks = np.zeros((len(order), self._park + 1), dtype=bool)
        for row, cid in enumerate(order):
            masks[row, :self._park] = self.flat_masks[cid].astype(bool)
        self._mask_t = torch.as_tensor(masks, device=device)
        self._area_t = torch.as_tensor(
            np.array([self.mask_area[cid] for cid in order], dtype=np.float32), device=device)
        self._buffer = torch.zeros((len(order), self._park + 1), dtype=torch.bool,
                                   device=device)
        self._body_t = torch.as_tensor(self.body_pts, dtype=torch.float32, device=device)
        self._lid_t = torch.as_tensor(self.lid_pts, dtype=torch.float32, device=device)
        return None

    def _world_points(self, pose_body: np.ndarray, theta: float) -> np.ndarray:
        lid_world = pose_body @ self.articulation.joint_transform(theta)
        return np.vstack([
            trimesh.transform_points(self.body_pts, pose_body),
            trimesh.transform_points(self.lid_pts, lid_world),
        ])

    def iou(self, pose_body: np.ndarray, theta: float) -> float:
        self.evaluations += 1
        if self.backend == "torch":
            return self._iou_torch(pose_body, theta)
        return self._iou_numpy(pose_body, theta)

    def _iou_numpy(self, pose_body: np.ndarray, theta: float) -> float:
        points = self._world_points(pose_body, theta)
        total = 0.0
        for cid, camera in self.cameras.items():
            flat_mask = self.flat_masks[cid]
            uv = camera.project(points)
            good = np.isfinite(uv).all(axis=1)
            u = np.rint(uv[good, 0]).astype(np.int32)
            v = np.rint(uv[good, 1]).astype(np.int32)
            inside = (u >= 0) & (u < camera.width - 1) & (v >= 0) & (v < camera.height - 1)
            u, v = u[inside], v[inside]
            if u.size == 0:
                continue
            # Splat each point as a 2x2 block: the samples are sparser than the
            # silhouette they stand for, so a bare point set understates overlap.
            base = v * camera.width + u
            hit = np.concatenate([base, base + 1, base + camera.width, base + camera.width + 1])
            # Deduplicate: area means distinct pixels. Leaving duplicates in would
            # inflate the predicted area by an amount that varies with how
            # compactly the candidate projects, which biases the objective.
            #
            # Without sorting: stamp each hit's position into the scratch buffer,
            # then ask which hits still see their own stamp. Repeated pixels collide
            # and exactly one writer survives, whichever it is, so `first` marks one
            # hit per distinct pixel. Four linear passes instead of an n log n sort.
            scratch = self._scratch[cid]
            order = np.arange(hit.size)
            scratch[hit] = order
            first = scratch[hit] == order
            scratch[hit] = -1                       # clear only what was touched
            covered = int(np.count_nonzero(first))
            intersection = int(np.count_nonzero(first & flat_mask[hit].astype(bool)))
            union = covered + self.mask_area[cid] - intersection
            if union:
                total += intersection / union
        return total / len(self.cameras)

    def _iou_torch(self, pose_body: np.ndarray, theta: float) -> float:
        torch = self._torch
        lid_world = pose_body @ self.articulation.joint_transform(theta)
        with torch.no_grad():
            body = torch.as_tensor(pose_body, dtype=torch.float32,
                                   device=self._projection.device)
            lid = torch.as_tensor(lid_world, dtype=torch.float32,
                                  device=self._projection.device)
            points = torch.cat([self._body_t @ body[:3, :3].T + body[:3, 3],
                                self._lid_t @ lid[:3, :3].T + lid[:3, 3]])
            camera = (torch.einsum("vij,nj->vni", self._projection[:, :, :3], points)
                      + self._projection[:, :, 3][:, None, :])
            depth = camera[..., 2]
            uv = camera[..., :2] / depth.unsqueeze(-1)
            u = torch.round(uv[..., 0]).long()
            v = torch.round(uv[..., 1]).long()
            good = (depth > 1e-6) & (u >= 0) & (u < self._width - 1) \
                & (v >= 0) & (v < self._height - 1)
            # Fold the rejects to the origin before the arithmetic: a point behind
            # the camera projects to +-inf, and int64 of that is a value whose
            # products overflow. `good` discards them either way, but the overflow
            # would happen first.
            u = torch.where(good, u, self._zero_t)
            v = torch.where(good, v, self._zero_t)
            base = v * self._width + u
            self._buffer.zero_()
            for offset in (0, 1, self._width, self._width + 1):
                self._buffer.scatter_(1, torch.where(good, base + offset, self._park_t), True)
            # The park column is False in the mask, so it can never join the
            # intersection; it is only removed from the covered area.
            covered = self._buffer.sum(1).float() - self._buffer[:, self._park].float()
            intersection = (self._buffer & self._mask_t).sum(1).float()
            union = covered + self._area_t - intersection
            return float((intersection / union.clamp_min(1.0)).mean())

    def cost(self, vector: np.ndarray, theta: float) -> float:
        return -self.iou(_to_pose(vector), theta)


def _to_pose(vector: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    pose[:3, 3] = vector[3:]
    return pose


def _from_pose(pose: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]])


def _refine(objective: SilhouetteObjective, vector: np.ndarray, theta: float,
            iterations: int) -> tuple[np.ndarray, float]:
    result = minimize(objective.cost, vector, args=(theta,), method="Powell",
                      options={"maxiter": iterations, "xtol": 1e-3, "ftol": 1e-4})
    return result.x, -result.fun


def fit(objective: SilhouetteObjective, seed_pose: np.ndarray,
        theta_min: float, theta_max: float,
        coarse_step_deg: float = 15.0, fine_step_deg: float = 2.0,
        refine_iterations: int = 8) -> dict:
    """Sweep theta, refine the body pose at each, then polish theta."""
    vector = _from_pose(seed_pose)

    coarse = theta_grid(theta_min, theta_max, coarse_step_deg)
    best = (-1.0, vector, 0.0)
    for theta in coarse:
        candidate, score = _refine(objective, vector, float(theta), refine_iterations)
        if score > best[0]:
            best = (score, candidate, float(theta))
    score, vector, theta = best

    fine = np.radians(np.arange(
        max(np.degrees(theta_min), np.degrees(theta) - coarse_step_deg),
        min(np.degrees(theta_max), np.degrees(theta) + coarse_step_deg) + 1e-6,
        fine_step_deg))
    for candidate_theta in fine:
        value = objective.iou(_to_pose(vector), float(candidate_theta))
        if value > score:
            score, theta = value, float(candidate_theta)

    vector, score = _refine(objective, vector, theta, refine_iterations * 2)
    return {"pose_body": _to_pose(vector), "theta": theta, "iou": score}


def _perturb(pose: np.ndarray, rot_deg: float, trans_mm: float,
             rng: np.random.Generator) -> np.ndarray:
    noisy = pose.copy()
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    noisy[:3, :3] = Rotation.from_rotvec(axis * np.radians(rot_deg)).as_matrix() @ pose[:3, :3]
    direction = rng.normal(size=3)
    noisy[:3, 3] = pose[:3, 3] + direction / np.linalg.norm(direction) * (trans_mm / 1000.0)
    return noisy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=Path(__file__).parent / "synthetic")
    parser.add_argument("--samples", type=int, default=2500, help="Surface points per part.")
    parser.add_argument("--seed-noise", type=float, nargs=2, metavar=("ROT_DEG", "TRANS_MM"),
                        default=[15.0, 30.0], help="Perturbation applied to the seed pose.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    truth = json.loads((args.synthetic / "ground_truth.json").read_text(encoding="utf-8"))
    articulation = load_articulation(truth["object"])
    cameras = {cid: cam.scaled(truth["render_scale"])
               for cid, cam in load_cameras(Path(truth["episode"])).items()}
    pose_true = np.asarray(truth["pose_body"], dtype=np.float64)
    rng = np.random.default_rng(args.seed)

    rot_deg, trans_mm = args.seed_noise
    print(f"seed noise: {rot_deg} deg / {trans_mm} mm   trials={args.trials}   "
          f"samples={args.samples}/part", flush=True)
    print(f"{'theta_gt':>9} {'trial':>5} {'theta_err':>10} {'rot_err':>8} {'trans_err':>10} "
          f"{'IoU':>6} {'evals':>7} {'sec':>6}", flush=True)

    rows = []
    for frame in truth["frames"]:
        frame_dir = args.synthetic / f"frame_{frame['frame']:03d}"
        masks = {cid: np.load(frame_dir / f"{cid}_parts.npy") >= 0 for cid in truth["camera_ids"]}
        objective = SilhouetteObjective(articulation, cameras, masks, args.samples, args.seed)

        for trial in range(args.trials):
            seed_pose = _perturb(pose_true, rot_deg, trans_mm, rng)
            objective.evaluations = 0
            started = time.perf_counter()
            out = fit(objective, seed_pose, articulation.theta_min, articulation.theta_max)
            elapsed = time.perf_counter() - started

            theta_err = abs(np.degrees(out["theta"] - frame["theta_rad"]))
            delta = np.linalg.inv(pose_true) @ out["pose_body"]
            rot_err = np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))
            trans_err = np.linalg.norm(delta[:3, 3]) * 1000.0
            rows.append({"theta_gt_deg": frame["theta_deg"], "trial": trial,
                         "theta_err_deg": theta_err, "rot_err_deg": rot_err,
                         "trans_err_mm": trans_err, "iou": out["iou"]})
            print(f"{frame['theta_deg']:9.1f} {trial:5d} {theta_err:9.2f}d {rot_err:7.2f}d "
                  f"{trans_err:9.2f}mm {out['iou']:6.3f} {objective.evaluations:7d} {elapsed:6.1f}",
                  flush=True)

    (args.synthetic / "fit_rc_results.json").write_text(
        json.dumps({"seed_noise": args.seed_noise, "rows": rows}, indent=2) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
