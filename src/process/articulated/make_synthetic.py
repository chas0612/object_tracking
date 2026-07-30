#!/usr/bin/env python3
"""Render a synthetic articulated sequence from the episode's real 22 cameras.

Produces, for each articulation angle, per-camera depth, part masks and a shaded
RGB image, plus the ground-truth ``(pose_body, theta)`` that produced them. This
is the fixture both candidate initialisers get measured on, so that the choice
between them rests on a measurement rather than on an argument.

Rendering is **ray casting on the CPU** (Embree, via open3d's tensor API), not
rasterisation. Two reasons:

* It uses no GPU, so it cannot contend with the allegro_v5 tracking campaign
  sharing this box's 4080 SUPER.
* Geometry IDs come back per pixel, so part masks are exact -- no antialiasing
  blend to threshold, and occlusion between lid and body is handled for free.

The shaded RGB is Lambertian over the ray-cast normals. It is good enough to
exercise geometry, but it carries none of the object's real texture, so it does
not test how an RGB-driven method copes with appearance.

Depth is written for only ``--depth-cameras`` views (default 1). The rig has
essentially one usable stereo pair, so a probe that helped itself to depth in all
22 views would be measuring a sensor we do not have.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from common import (
    BODY,
    DEFAULT_EPISODE,
    DEFAULT_OBJECT,
    LID,
    Articulation,
    Camera,
    load_articulation,
    load_cameras,
    reference_pose,
)

MISS = np.float32(np.inf)
LIGHT_DIR = np.array([0.3, 0.4, 0.87])          # fixed key light, world frame
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
PART_ALBEDO = {BODY: np.array([0.30, 0.55, 0.30]), LID: np.array([0.35, 0.55, 0.85])}


def _scene(articulation: Articulation, pose_body: np.ndarray, theta: float):
    """A raycasting scene holding the two parts, body first so ids are BODY/LID."""
    body_v, lid_v = articulation.posed(pose_body, theta)
    scene = o3d.t.geometry.RaycastingScene()
    for vertices, faces in ((body_v, articulation.body.faces), (lid_v, articulation.lid.faces)):
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
            o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.uint32)),
        )
        scene.add_triangles(mesh)
    return scene


def _render(scene, camera: Camera) -> dict[str, np.ndarray]:
    """Cast one ray per pixel. Returns depth (m, 0 where missed), part ids, RGB."""
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(camera.K),
        extrinsic_matrix=o3d.core.Tensor(camera.extrinsic),
        width_px=camera.width,
        height_px=camera.height,
    )
    hit = scene.cast_rays(rays)

    t_hit = hit["t_hit"].numpy()
    valid = np.isfinite(t_hit)
    geometry = hit["geometry_ids"].numpy()
    normals = hit["primitive_normals"].numpy()

    rays_np = rays.numpy()
    points = rays_np[..., :3] + t_hit[..., None] * rays_np[..., 3:]
    camera_points = trimesh.transform_points(points.reshape(-1, 3), camera.extrinsic)
    depth = camera_points[:, 2].reshape(t_hit.shape).astype(np.float32)
    depth[~valid] = 0.0

    parts = np.full(t_hit.shape, -1, dtype=np.int8)
    parts[valid] = geometry[valid].astype(np.int8)

    # Lambertian, two-sided so back-facing scan normals do not read as holes.
    shade = np.abs(normals @ LIGHT_DIR)
    rgb = np.zeros(t_hit.shape + (3,), dtype=np.float32)
    for part, albedo in PART_ALBEDO.items():
        sel = parts == part
        rgb[sel] = albedo * (0.25 + 0.75 * shade[sel])[:, None]
    return {
        "depth": depth,
        "parts": parts,
        "rgb": (np.clip(rgb, 0, 1) * 255).astype(np.uint8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default=DEFAULT_OBJECT)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--thetas-deg", type=float, nargs="*", default=[0.0, 45.0, 90.0, 135.0, 180.0])
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Render resolution as a fraction of the real 2048x1536.")
    parser.add_argument("--depth-cameras", type=int, default=1,
                        help="How many views get a depth map. The rig has ~1 stereo pair; "
                             "0 means all, which is a deliberate upper bound, not the real rig.")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic")
    args = parser.parse_args()

    articulation = load_articulation(args.object)
    cameras = {cid: cam.scaled(args.scale) for cid, cam in load_cameras(args.episode).items()}
    pose_body = reference_pose(args.episode)
    camera_ids = sorted(cameras)
    depth_ids = camera_ids if args.depth_cameras <= 0 else camera_ids[: args.depth_cameras]

    print(f"object={args.object} cameras={len(cameras)} "
          f"resolution={cameras[camera_ids[0]].width}x{cameras[camera_ids[0]].height} "
          f"depth_views={len(depth_ids)} thetas={args.thetas_deg}", flush=True)

    # Sanity: the object must actually land inside the images we are about to render.
    centre = trimesh.transform_points(
        np.atleast_2d(articulation.body.centroid), pose_body)
    visible = 0
    for cid in camera_ids:
        uv = cameras[cid].project(centre)[0]
        if np.all(np.isfinite(uv)) and 0 <= uv[0] < cameras[cid].width and 0 <= uv[1] < cameras[cid].height:
            visible += 1
    print(f"object centre projects inside {visible}/{len(camera_ids)} views", flush=True)
    if visible == 0:
        raise ValueError("Object centre is outside every view; check the extrinsic convention.")

    args.out.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, theta_deg in enumerate(args.thetas_deg):
        theta = float(np.radians(theta_deg))
        scene = _scene(articulation, pose_body, theta)
        frame_dir = args.out / f"frame_{index:03d}"
        frame_dir.mkdir(exist_ok=True)

        coverage = {}
        for cid in camera_ids:
            out = _render(scene, cameras[cid])
            np.save(frame_dir / f"{cid}_parts.npy", out["parts"])
            np.save(frame_dir / f"{cid}_rgb.npy", out["rgb"])
            if cid in depth_ids:
                np.save(frame_dir / f"{cid}_depth.npy", out["depth"])
            coverage[cid] = {
                "body_px": int((out["parts"] == BODY).sum()),
                "lid_px": int((out["parts"] == LID).sum()),
            }

        lid_seen = sum(1 for v in coverage.values() if v["lid_px"] > 50)
        frames.append({
            "frame": index,
            "theta_deg": theta_deg,
            "theta_rad": theta,
            "lid_visible_views": lid_seen,
            "coverage": coverage,
        })
        print(f"  theta={theta_deg:6.1f}deg  lid visible in {lid_seen}/{len(camera_ids)} views",
              flush=True)

    (args.out / "ground_truth.json").write_text(json.dumps({
        "object": args.object,
        "episode": str(args.episode),
        "pose_body": pose_body.tolist(),
        "joint_axis": articulation.axis.tolist(),
        "joint_origin": articulation.origin.tolist(),
        "theta_max_rad": articulation.theta_max,
        "render_scale": args.scale,
        "camera_ids": camera_ids,
        "depth_camera_ids": depth_ids,
        "frames": frames,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
