#!/usr/bin/env python3
"""First 7-DoF pose on real footage, from multi-view silhouettes alone.

Same objective as ``fit_rc.py``, but on a real capture, where there is no ground
truth to seed from. The seed is built from the data instead:

1. Triangulate the per-view mask centroids to locate the object. Silhouette
   centroids are not the projection of the 3D centroid, so this is only a
   position estimate -- which is all it is used for.
2. Search orientation globally: a fixed quasi-random set of rotations crossed
   with the theta grid, scored by silhouette IoU. Cheap because each score is
   just a projection of surface samples.
3. Refine the best few candidates properly.

Orientation is searched rather than derived from a hull PCA because the object is
articulated: the hull's principal axes depend on how far the lid is open, so they
cannot be matched against the mesh's until theta is already known.

Writes an overlay sheet next to the result so the fit can be checked by eye,
which -- for a first pose with no ground truth -- is the only honest check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from common import Camera, load_articulation, load_cameras
from fit_rc import SilhouetteObjective, _from_pose, _refine, _to_pose

DEFAULT_CAPTURE = (Path.home() /
                   "shared_data/capture/eccv2026/capture_hand/right/box_articulated/0")


def _load_masks(mask_dir: Path, cameras: dict[str, Camera], scale: float
                ) -> tuple[dict[str, np.ndarray], dict[str, Camera]]:
    """Binary masks downscaled to the working resolution, with matching cameras."""
    masks, kept = {}, {}
    for path in sorted(mask_dir.glob("*.png")):
        camera_id = path.stem
        if camera_id not in cameras:
            continue
        camera = cameras[camera_id].scaled(scale)
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.sum() == 0:
            continue
        masks[camera_id] = cv2.resize(raw, (camera.width, camera.height),
                                      interpolation=cv2.INTER_AREA) > 127
        kept[camera_id] = camera
    return masks, kept


def _triangulate_centroids(masks: dict[str, np.ndarray], cameras: dict[str, Camera]
                           ) -> np.ndarray:
    """Least-squares intersection of the rays through each view's mask centroid."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for camera_id, mask in masks.items():
        camera = cameras[camera_id]
        ys, xs = np.nonzero(mask)
        uv = np.array([xs.mean(), ys.mean(), 1.0])
        direction_cam = np.linalg.inv(camera.K) @ uv
        world_from_cam = np.linalg.inv(camera.extrinsic)
        origin = world_from_cam[:3, 3]
        direction = world_from_cam[:3, :3] @ direction_cam
        direction /= np.linalg.norm(direction)
        projector = np.eye(3) - np.outer(direction, direction)
        A += projector
        b += projector @ origin
    return np.linalg.solve(A, b)


def _global_search(objective: SilhouetteObjective, centre: np.ndarray, theta_max: float,
                   rotations: int, theta_step_deg: float, seed: int,
                   mesh_centroid: np.ndarray) -> list[tuple[float, np.ndarray, float]]:
    """Score a rotation x theta sweep, coarsest possible, and rank the results."""
    orientations = Rotation.random(rotations, random_state=seed).as_matrix()
    thetas = np.radians(np.arange(0.0, np.degrees(theta_max) + 1e-6, theta_step_deg))
    scored = []
    for rotation in orientations:
        pose = np.eye(4)
        pose[:3, :3] = rotation
        # Put the model's own centroid on the triangulated centre.
        pose[:3, 3] = centre - rotation @ mesh_centroid
        for theta in thetas:
            scored.append((objective.iou(pose, float(theta)), pose, float(theta)))
    scored.sort(key=lambda item: -item[0])
    return scored


def _overlay_sheet(path: Path, images_dir: Path, masks: dict[str, np.ndarray],
                   cameras: dict[str, Camera], articulation, pose: np.ndarray,
                   theta: float, columns: int = 3, rows: int = 2) -> None:
    body, lid = articulation.posed(pose, theta)
    tiles = []
    for camera_id in list(masks)[: columns * rows]:
        camera = cameras[camera_id]
        image = cv2.imread(str(images_dir / f"{camera_id}.png"))
        image = cv2.resize(image, (camera.width, camera.height))
        for points, colour in ((body, (90, 220, 90)), (lid, (240, 160, 60))):
            uv = camera.project(points[:: max(1, len(points) // 6000)])
            good = np.isfinite(uv).all(axis=1)
            uv = uv[good].astype(np.int32)
            inside = ((uv[:, 0] >= 0) & (uv[:, 0] < camera.width)
                      & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height))
            image[uv[inside, 1], uv[inside, 0]] = colour
        tiles.append(image)
    grid = np.vstack([np.hstack(tiles[i * columns:(i + 1) * columns]) for i in range(rows)])
    cv2.imwrite(str(path), grid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--frame-index", type=int, default=40)
    parser.add_argument("--object", default="blue_plastic_box")
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--rotations", type=int, default=600)
    parser.add_argument("--theta-step-deg", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=8, help="Coarse candidates to refine.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: <capture-dir>/articulated_probe/frame_<index>/")
    args = parser.parse_args()

    frame_dir = args.capture_dir / f"foundpose_frame_{args.frame_index:06d}"
    out_dir = args.out or (args.capture_dir / "articulated_probe" / f"frame_{args.frame_index:06d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    articulation = load_articulation(args.object)
    masks, cameras = _load_masks(frame_dir / "masks", load_cameras(args.capture_dir), args.scale)
    if not masks:
        raise FileNotFoundError(f"No usable masks under {frame_dir / 'masks'}")
    width, height = next(iter(cameras.values())).width, next(iter(cameras.values())).height
    print(f"views={len(masks)}  {width}x{height}  rotations={args.rotations}  "
          f"theta_step={args.theta_step_deg}deg", flush=True)

    centre = _triangulate_centroids(masks, cameras)
    print(f"triangulated centre = {np.round(centre, 4)} m", flush=True)

    objective = SilhouetteObjective(articulation, cameras, masks, args.samples, args.seed)
    started = time.perf_counter()
    scored = _global_search(objective, centre, articulation.theta_max, args.rotations,
                            args.theta_step_deg, args.seed, articulation.body.centroid)
    print(f"coarse: {len(scored)} candidates in {time.perf_counter() - started:.1f}s   "
          f"best IoU {scored[0][0]:.3f}, {args.top_k}th {scored[args.top_k - 1][0]:.3f}",
          flush=True)

    best = (-1.0, None, 0.0)
    for rank, (_, pose, theta) in enumerate(scored[: args.top_k]):
        vector, score = _refine(objective, _from_pose(pose), theta, iterations=20)
        for candidate in np.radians(np.arange(max(0.0, np.degrees(theta) - args.theta_step_deg),
                                              min(np.degrees(articulation.theta_max),
                                                  np.degrees(theta) + args.theta_step_deg) + 1e-6,
                                              2.0)):
            value = objective.iou(_to_pose(vector), float(candidate))
            if value > score:
                score, theta = value, float(candidate)
        vector, score = _refine(objective, vector, theta, iterations=30)
        print(f"  candidate {rank}: IoU {score:.4f}  theta {np.degrees(theta):6.1f}deg", flush=True)
        if score > best[0]:
            best = (score, _to_pose(vector), theta)

    score, pose, theta = best
    elapsed = time.perf_counter() - started
    print(f"\nBEST  IoU={score:.4f}  theta={np.degrees(theta):.1f}deg  "
          f"({elapsed:.1f}s, {objective.evaluations} evaluations)", flush=True)
    print(f"pose_body =\n{np.round(pose, 4)}", flush=True)

    (out_dir / "first_pose.json").write_text(json.dumps({
        "capture_dir": str(args.capture_dir),
        "frame_index": args.frame_index,
        "object": args.object,
        "views_used": sorted(masks),
        "render_scale": args.scale,
        "pose_body": pose.tolist(),
        "theta_rad": theta,
        "theta_deg": float(np.degrees(theta)),
        "iou": score,
        "triangulated_centre": centre.tolist(),
    }, indent=2) + "\n", encoding="utf-8")

    _overlay_sheet(out_dir / "overlay.png", frame_dir / "images", masks, cameras,
                   articulation, pose, theta)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
