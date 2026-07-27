#!/usr/bin/env python3
"""Render a compact multiview sheet for one FoundPose initialization.

Each cell shows the bootstrap RGB image, the selected world-pose mesh overlay,
and the corresponding SAM mask boundary when that camera produced a mask.
This isolates initialization quality from subsequent GoTrack failures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.overlay_object_video_single import (  # noqa: E402
    ObjectOverlayRenderer,
    load_cam_param,
)


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in dict.fromkeys(indices)]


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh: {path}")
    return mesh


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 360), 30), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (7, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--foundpose-frame-dir", required=True)
    parser.add_argument("--init-pose", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-ids", nargs="*", default=None)
    parser.add_argument("--max-cameras", type=int, default=8)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cameras < 1 or args.columns < 1 or args.cell_width < 64:
        raise ValueError("camera count, columns, and cell width must be positive")
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("--alpha must be in (0, 1]")

    capture = Path(args.capture_dir).expanduser().resolve()
    mesh_path = Path(args.object_mesh).expanduser().resolve()
    frame_dir = Path(args.foundpose_frame_dir).expanduser().resolve()
    pose_path = Path(args.init_pose).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    image_dir = frame_dir / "images"
    mask_dir = frame_dir / "masks"

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for path in (capture, image_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (mesh_path, pose_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    intrinsics, extrinsics = load_cam_param(capture / "cam_param")
    image_ids = sorted(path.stem for path in image_dir.glob("*.png"))
    usable = [serial for serial in image_ids if serial in intrinsics and serial in extrinsics]
    mask_ids = {path.stem for path in mask_dir.glob("*.png")} if mask_dir.is_dir() else set()
    evidence_ids = [serial for serial in usable if serial in mask_ids]
    candidates = evidence_ids or usable
    serials = (
        list(args.camera_ids)
        if args.camera_ids is not None
        else _evenly_spaced(candidates, args.max_cameras)
    )
    invalid = [serial for serial in serials if serial not in usable]
    if invalid:
        raise ValueError(f"Unusable camera IDs: {invalid}")
    if not serials:
        raise RuntimeError("No calibrated bootstrap images")

    images = [cv2.imread(str(image_dir / f"{serial}.png"), cv2.IMREAD_COLOR) for serial in serials]
    if any(image is None for image in images):
        raise RuntimeError("Could not read one or more bootstrap images")
    height, width = images[0].shape[:2]
    if any(image.shape[:2] != (height, width) for image in images):
        raise ValueError("Bootstrap image dimensions differ across cameras")

    pose = np.asarray(np.load(pose_path), dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0, 0, 0, 1]])
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid world pose in {pose_path}: {pose.shape}")

    renderer = ObjectOverlayRenderer(
        _load_mesh(mesh_path),
        {serial: intrinsics[serial] for serial in serials},
        {serial: extrinsics[serial] for serial in serials},
        height,
        width,
        alpha=args.alpha,
    )
    overlays = renderer.render(pose, images)
    cell_height = round(height * args.cell_width / width)
    cells: list[np.ndarray] = []
    for serial, overlay in zip(renderer.serials, overlays):
        mask_path = mask_dir / f"{serial}.png"
        if mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (height, width):
                raise ValueError(f"Invalid mask: {mask_path}")
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3, cv2.LINE_AA)
            mask_status = "SAM=yes"
        else:
            mask_status = "SAM=no"
        resized = cv2.resize(
            overlay,
            (args.cell_width, cell_height),
            interpolation=cv2.INTER_AREA,
        )
        cells.append(_label(resized, f"{serial}  {mask_status}"))

    columns = min(args.columns, len(cells))
    rows: list[np.ndarray] = []
    blank = np.zeros_like(cells[0])
    for start in range(0, len(cells), columns):
        row = cells[start : start + columns]
        row.extend(blank.copy() for _ in range(columns - len(row)))
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    if not cv2.imwrite(str(output), sheet, params):
        raise RuntimeError(f"Could not write {output}")
    print(
        f"[foundpose-init-sheet] cameras={serials} masks={len(evidence_ids)}/{len(usable)} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
