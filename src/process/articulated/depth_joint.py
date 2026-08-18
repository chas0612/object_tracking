"""Score a joint coordinate against measured stereo depth, without a mask.

Why this exists at all. The seed ranks its hypotheses by multi-view silhouette IoU,
and that works whenever the mask contains the whole object. On drawer/2 it did not:
SAM3 given "white box" returned the cabinet without the extended drawer, and a mask
holding only the cabinet is best explained by a **shut** drawer. The fit maximised
correctly against the wrong target and returned 0.6 mm where the truth was ~220.
Nothing failed loudly; the absolute IoU (0.32, against 0.68 on scissors) was the only
tell.

The fix is not a better mask. It is to stop asking the mask about the joint. Stereo
depth is computed over the whole image -- on that frame 200-243k valid pixels per
pair, of which only 12-33k lie inside the mask -- so the extended drawer's surface is
measured whether or not anything segmented it. Scoring the *moving part's visible
pixels* against that measurement needs no mask anywhere, and separates the truth from
the shut answer by a factor of two:

    joint       0 mm   120 mm   180 mm   220 mm   240 mm   280 mm   320 mm
    agreement  0.329    0.452    0.576    0.671    0.673    0.632    0.604

That is also how the drawer's declared range was caught being short: `joint.json`
gives 207.9 mm as "convention, not measured", and the peak sits past it at 220-240.

The division of labour this implies, and which the caller enforces: **depth decides
the joint coordinate, the silhouette decides the body.** An outline barely constrains
distance along the viewing ray, so the body stays where depth-based registration put
it and the silhouette only refines it; and a silhouette cut from an incomplete mask
cannot see a sliding part at all, so the joint is depth's to answer.

Nothing here is revolute-specific or prismatic-specific -- it scores a joint
coordinate of either kind. It is only *used* on prismatic joints, because that is
where the silhouette was measured to fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


class DepthJointObjective:
    """Fraction of the moving part's visible pixels whose depth the measurement agrees with.

    Visibility is resolved with a z-buffer over surface samples of *both* parts, so a
    drawer still inside the cabinet is correctly reported as hidden rather than scored
    against the cabinet's own front face. Samples rather than triangles because this is
    evaluated a few hundred times per seed: a triangle rasteriser over this mesh's
    227k faces took ~8 s per camera per evaluation, and splatting a fixed number of
    surface points is ~10 ms with the same answer to within a percent.
    """

    def __init__(self, articulation, manifest: Path, intrinsics: dict, extrinsics: dict,
                 scale: float = 0.25, tolerance_m: float = 0.02,
                 samples: int = 120000, splat: int = 1, seed: int = 0):
        self.articulation = articulation
        self.tolerance_m = float(tolerance_m)
        self.splat = int(splat)

        rng = np.random.default_rng(seed)
        # Split the sample budget by surface area so neither part is under-drawn: an
        # under-drawn body lets the drawer show through it and a hidden drawer scores
        # as if it were visible.
        areas = np.array([float(articulation.body.area), float(articulation.lid.area)])
        counts = np.maximum((samples * areas / areas.sum()).astype(int), 1000)
        points, part_ids = [], []
        for part_id, (mesh, count) in enumerate(
                zip((articulation.body, articulation.lid), counts)):
            sampled, _ = trimesh.sample.sample_surface(mesh, int(count), seed=int(rng.integers(1 << 30)))
            points.append(np.asarray(sampled, dtype=np.float64))
            part_ids.append(np.full(len(sampled), part_id, dtype=np.int8))
        self.points = np.vstack(points)
        self.part_ids = np.concatenate(part_ids)
        self.moving = self.part_ids == 1

        self.views = []
        for spec, entry in json.loads(Path(manifest).read_text(encoding="utf-8"))["pairs"].items():
            camera_id = entry["reference_camera"]
            depth = np.load(entry["depth_npy"])
            height = int(round(depth.shape[0] * scale))
            width = int(round(depth.shape[1] * scale))
            K = np.asarray(intrinsics[camera_id]["intrinsics_undistort"],
                           dtype=np.float64).reshape(3, 3).copy()
            K[:2, :] *= width / float(intrinsics[camera_id]["width"])
            extrinsic = np.eye(4)
            extrinsic[:3, :4] = np.asarray(extrinsics[camera_id], dtype=np.float64)
            self.views.append({
                "spec": spec,
                "camera_id": camera_id,
                # INTER_NEAREST: depth is a measurement, not an image. Averaging a
                # foreground pixel with a background one invents a surface between them.
                "depth": cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST),
                "K": K,
                "extrinsic": extrinsic,
                "width": width,
                "height": height,
            })
        if not self.views:
            raise ValueError(f"{manifest}: no stereo pairs to score against")

    def _visible_moving(self, view: dict, pose_body: np.ndarray, joint_value: float):
        """(pixel index, predicted depth) where the moving part is the nearest surface."""
        transform = self.articulation.joint_transform(float(joint_value))
        world = self.points.copy()
        world[self.moving] = (self.points[self.moving] @ transform[:3, :3].T
                              + transform[:3, 3])
        camera = trimesh.transform_points(world, view["extrinsic"] @ pose_body)
        z = camera[:, 2]
        uv = (view["K"] @ camera.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        u = np.rint(uv[:, 0]).astype(np.int64)
        v = np.rint(uv[:, 1]).astype(np.int64)
        keep = (z > 1e-6) & (u >= 0) & (u < view["width"]) & (v >= 0) & (v < view["height"])
        if not keep.any():
            return None, None
        u, v, z = u[keep], v[keep], z[keep]
        moving = self.moving[keep]

        # Painter's z-buffer: write far first so the nearest sample of either part
        # ends up owning the pixel. Sorting is what makes this exact rather than
        # approximate -- np.minimum.at would give the right depth but not tell us
        # which part won it.
        order = np.argsort(-z, kind="stable")
        u, v, z, moving = u[order], v[order], z[order], moving[order]
        flat = v * view["width"] + u
        size = view["width"] * view["height"]
        depth_buffer = np.full(size, np.inf, dtype=np.float64)
        moving_buffer = np.zeros(size, dtype=bool)
        for du in range(-self.splat, self.splat + 1):
            for dv in range(-self.splat, self.splat + 1):
                shifted = flat + dv * view["width"] + du
                inside = (u + du >= 0) & (u + du < view["width"]) \
                    & (v + dv >= 0) & (v + dv < view["height"])
                index = shifted[inside]
                depth_buffer[index] = z[inside]
                moving_buffer[index] = moving[inside]
        pixels = np.nonzero(moving_buffer)[0]
        return pixels, depth_buffer[pixels]

    def agreement(self, pose_body: np.ndarray, joint_value: float) -> float:
        """Agreeing pixels / measured pixels, pooled over every stereo pair.

        Pooled rather than averaged per view so that a pair which happens to see very
        little of the moving part cannot carry the same weight as one that sees all
        of it. Pixels with no measurement are excluded rather than counted as
        disagreement: stereo drops out on textureless white, and treating a dropout
        as evidence against a hypothesis would prefer whichever pose hides the part.
        """
        agree = total = 0
        for view in self.views:
            pixels, predicted = self._visible_moving(view, np.asarray(pose_body, dtype=np.float64),
                                                     joint_value)
            if pixels is None:
                continue
            measured = view["depth"].reshape(-1)[pixels]
            valid = measured > 0
            if not valid.any():
                continue
            agree += int((np.abs(measured[valid] - predicted[valid]) < self.tolerance_m).sum())
            total += int(valid.sum())
        return agree / total if total else 0.0

    def visible_pixels(self, pose_body: np.ndarray, joint_value: float) -> int:
        """How many measured pixels the score above was computed from.

        Reported alongside the score because the two move together as a part slides
        out of frame, and a rising score on a collapsing sample is not an improvement.
        """
        total = 0
        for view in self.views:
            pixels, _ = self._visible_moving(view, np.asarray(pose_body, dtype=np.float64),
                                             joint_value)
            if pixels is None:
                continue
            total += int((view["depth"].reshape(-1)[pixels] > 0).sum())
        return total
