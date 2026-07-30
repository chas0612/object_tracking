#!/usr/bin/env python3
"""Composite the posed articulated mesh onto the real images, for eyeballing a fit.

The point-splat sheets that `real_first_pose._overlay_sheet` draws are fine for
confirming an answer is in the right place, but they cannot show whether it is
*right*: scattered points have no occlusion, so the lid's far face shows through
the body and a silhouette that overshoots by ten pixels looks the same as one
that does not.

So raycast instead, the same way `make_synthetic` does. That gives a depth buffer
-- hidden surfaces stay hidden -- and a per-pixel part id, so body and lid get
their own colour and their own outline. The outline is what to look at: a correct
pose puts it exactly on the object's edge in the photograph.

Rendering runs on the CPU, at whatever resolution the caller asks for. It is not
fast, but it is a handful of frames at the end of a run and it never contends
with a GPU that other work may be using.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import BODY, LID, Articulation, Camera, load_articulation, load_cameras  # noqa: E402
from make_synthetic import _render, _scene  # noqa: E402

# BGR. Body cool, lid warm, so the two parts never read as one blob.
PART_COLOUR = {BODY: (90, 230, 120), LID: (70, 140, 255)}
FILL_ALPHA = 0.40
PAD_PX = 45


def shared_window(camera: Camera, articulation: Articulation,
                  answers: list[tuple[np.ndarray, float]]) -> tuple[int, int, int, int]:
    """A crop box covering every candidate answer, so stacked rows stay comparable.

    Cropping each row to its own mesh silently re-centres it, which is exactly the
    displacement the sheet exists to show. Projected vertices are enough to size
    the box -- no need to raycast twice.
    """
    corners = []
    for pose, theta in answers:
        body, lid = articulation.posed(pose, theta)
        uv = camera.project(np.vstack([body, lid]))
        uv = uv[np.isfinite(uv).all(axis=1)]
        if uv.size:
            corners.append(uv)
    if not corners:
        return 0, 0, camera.width, camera.height
    uv = np.vstack(corners)
    x0, y0 = np.floor(uv.min(0)).astype(int) - PAD_PX
    x1, y1 = np.ceil(uv.max(0)).astype(int) + PAD_PX
    return (max(0, int(x0)), max(0, int(y0)),
            min(camera.width, int(x1)), min(camera.height, int(y1)))


def mesh_overlay(image: np.ndarray, camera: Camera, articulation: Articulation,
                 pose: np.ndarray, theta: float, crop: bool = True,
                 window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """One view: the photograph with the posed mesh shaded over it and outlined."""
    rendered = _render(_scene(articulation, pose, theta), camera)
    parts = rendered["parts"]
    canvas = cv2.resize(image, (camera.width, camera.height)).astype(np.float32)

    for part, colour in PART_COLOUR.items():
        region = parts == part
        if not region.any():
            continue
        # Shade by the render's own Lambertian term so curvature stays legible
        # through the blend, rather than flooding the part with a flat colour.
        shade = rendered["rgb"][region].max(axis=1)[:, None] / 255.0
        tint = np.asarray(colour, dtype=np.float32) * (0.45 + 0.55 * shade)
        canvas[region] = (1 - FILL_ALPHA) * canvas[region] + FILL_ALPHA * tint

    for part, colour in PART_COLOUR.items():
        mask = (parts == part).astype(np.uint8)
        if not mask.any():
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, colour, 2)

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    if window is not None:
        x0, y0, x1, y1 = window
        return canvas[y0:y1, x0:x1]
    if not crop:
        return canvas
    ys, xs = np.nonzero(parts >= 0)
    if xs.size == 0:
        return canvas
    x0, x1 = max(0, xs.min() - PAD_PX), min(camera.width, xs.max() + PAD_PX)
    y0, y1 = max(0, ys.min() - PAD_PX), min(camera.height, ys.max() + PAD_PX)
    return canvas[y0:y1, x0:x1]


def pick_views(cameras: dict[str, Camera], count: int) -> list[str]:
    """An even spread through the serial list -- deterministic, and not all one wall."""
    names = sorted(cameras)
    if len(names) <= count:
        return names
    return [names[int(round(i * (len(names) - 1) / (count - 1)))] for i in range(count)]


def mesh_overlay_sheet(path: Path, images_dir: Path, cameras: dict[str, Camera],
                       articulation: Articulation, pose: np.ndarray, theta: float,
                       label: str = "", views: list[str] | None = None,
                       tile: tuple[int, int] = (480, 360),
                       windows: dict[str, tuple[int, int, int, int]] | None = None,
                       ) -> list[str]:
    """A row of views with the mesh composited. Returns the views actually drawn."""
    views = views or pick_views(cameras, 5)
    drawn, tiles = [], []
    for camera_id in views:
        image = cv2.imread(str(images_dir / f"{camera_id}.png"))
        if image is None:
            continue
        panel = mesh_overlay(image, cameras[camera_id], articulation, pose, theta,
                             window=(windows or {}).get(camera_id))
        panel = cv2.resize(panel, tile)
        cv2.putText(panel, camera_id, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        tiles.append(panel)
        drawn.append(camera_id)
    if not tiles:
        return []
    sheet = np.hstack(tiles)
    if label:
        cv2.putText(sheet, label, (8, tile[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (0, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    return drawn


def main() -> int:
    """Re-render the sheets for a finished run, without repeating the fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--object", default="blue_plastic_box")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Render resolution as a fraction of the calibrated size.")
    parser.add_argument("--views", nargs="*", default=None)
    args = parser.parse_args()

    probe = (args.capture_dir / "articulated_probe" / f"frame_{args.frame_index:06d}")
    out_dir = probe / "hybrid"
    results = json.loads((out_dir / "hybrid_result.json").read_text())["results"]
    fit_path = out_dir / "depth_fit.json"
    if fit_path.is_file():
        # Every seed converges to the same place, so any of them is the answer.
        seed = min(json.loads(fit_path.read_text())["seeds"].values(),
                   key=lambda s: s["median_mm"])
        results["depth-fit"] = {"pose_body": seed["pose_body"],
                                "theta_deg": seed["theta_deg"],
                                "reference_camera": "depthfit",
                                "silhouette_iou": float("nan")}
    articulation = load_articulation(args.object)
    cameras = {cid: cam.scaled(args.scale)
               for cid, cam in load_cameras(args.capture_dir).items()}
    images = args.capture_dir / f"foundpose_frame_{args.frame_index:06d}" / "images"

    views = args.views or pick_views(cameras, 5)
    answers = [(np.asarray(r["pose_body"]), np.radians(r["theta_deg"]))
               for r in results.values()]
    windows = {cid: shared_window(cameras[cid], articulation, answers) for cid in views}

    rows = []
    for name, result in results.items():
        label = (f"{name}  theta={result['theta_deg']:.1f}deg  "
                 f"IoU={result['silhouette_iou']:.4f}")
        path = out_dir / f"final_{result['reference_camera']}.png"
        mesh_overlay_sheet(path, images, cameras, articulation,
                           np.asarray(result["pose_body"]),
                           np.radians(result["theta_deg"]), label, views,
                           windows=windows)
        print(f"wrote {path}")
        rows.append(cv2.imread(str(path)))
    if len(rows) > 1:
        stacked = out_dir / "final_all_pairs.png"
        cv2.imwrite(str(stacked), np.vstack(rows))
        print(f"wrote {stacked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
