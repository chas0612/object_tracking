#!/usr/bin/env python3
"""Transfer an Artec multi-texture PLY atlas onto a lightweight mesh.

Artec binary PLY exports store geometry faces and texture-wedge faces as
separate elements.  Some MeshLab decimation/export paths discard those UVs
while leaving the texture PNG beside an untextured OBJ.  This tool samples the
raw wedge UV at the nearest raw vertex and writes vertex colours onto the
lightweight geometry, preserving a small renderable mesh for FoundPose.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _header(path: Path) -> tuple[int, dict[str, int]]:
    with path.open("rb") as stream:
        block = stream.read(16384)
    marker = b"end_header\n"
    end = block.index(marker) + len(marker)
    text = block[:end].decode("ascii")
    counts = {
        name: int(count)
        for name, count in re.findall(r"^element (\S+) (\d+)$", text, re.MULTILINE)
    }
    required = {"vertex", "face", "multi_texture_vertex", "multi_texture_face"}
    if not required.issubset(counts):
        raise ValueError(f"Not a supported Artec multi-texture PLY: {path}")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("Only binary_little_endian PLY is supported")
    return end, counts


def _raw_arrays(path: Path):
    offset, counts = _header(path)
    vertex_count = counts["vertex"]
    face_count = counts["face"]
    texture_vertex_count = counts["multi_texture_vertex"]
    texture_face_count = counts["multi_texture_face"]
    if texture_face_count != face_count:
        raise ValueError("Geometry and texture face counts differ")

    vertices = np.memmap(path, dtype="<f4", mode="r", offset=offset,
                         shape=(vertex_count, 3))
    face_dtype = np.dtype([("count", "u1"), ("vertex", "<i4", (3,))], align=False)
    face_offset = offset + vertex_count * 3 * 4
    faces = np.memmap(path, dtype=face_dtype, mode="r", offset=face_offset,
                      shape=face_count)
    if not np.all(faces["count"] == 3):
        raise ValueError("Only triangular Artec PLY faces are supported")

    texture_vertex_dtype = np.dtype(
        [("texture", "u1"), ("uv", "<f4", (2,))], align=False,
    )
    texture_vertex_offset = face_offset + face_count * face_dtype.itemsize
    texture_vertices = np.memmap(
        path, dtype=texture_vertex_dtype, mode="r", offset=texture_vertex_offset,
        shape=texture_vertex_count,
    )
    texture_face_dtype = np.dtype(
        [("texture", "u1"), ("number", "<u4"), ("count", "u1"),
         ("vertex", "<i4", (3,))], align=False,
    )
    texture_face_offset = (
        texture_vertex_offset + texture_vertex_count * texture_vertex_dtype.itemsize
    )
    texture_faces = np.memmap(
        path, dtype=texture_face_dtype, mode="r", offset=texture_face_offset,
        shape=texture_face_count,
    )
    expected_size = texture_face_offset + texture_face_count * texture_face_dtype.itemsize
    if expected_size != path.stat().st_size or not np.all(texture_faces["count"] == 3):
        raise ValueError("Unexpected Artec PLY binary layout")
    if np.any(texture_vertices["texture"] != 0) or np.any(texture_faces["texture"] != 0):
        raise ValueError("Multiple texture atlases are not currently supported")
    return vertices, faces["vertex"], texture_vertices["uv"], texture_faces["vertex"]


def _representative_uv_per_vertex(
    vertex_count: int, faces: np.ndarray, texture_faces: np.ndarray,
    chunk_faces: int = 250_000,
) -> np.ndarray:
    texture_index = np.full(vertex_count, -1, dtype=np.int32)
    for begin in range(0, len(faces), chunk_faces):
        end = min(len(faces), begin + chunk_faces)
        geometry = np.asarray(faces[begin:end])
        texture = np.asarray(texture_faces[begin:end])
        for corner in range(3):
            vertices = geometry[:, corner]
            texture_vertices = texture[:, corner]
            missing = texture_index[vertices] < 0
            texture_index[vertices[missing]] = texture_vertices[missing]
    if np.any(texture_index < 0):
        missing = int(np.sum(texture_index < 0))
        raise RuntimeError(f"{missing} raw vertices have no texture wedge")
    return texture_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ply", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--geometry-mesh", required=True)
    parser.add_argument("--output", required=True, help="New vertex-coloured .ply")
    parser.add_argument("--raw-scale", type=float, default=0.001,
                        help="Raw PLY unit to geometry-mesh unit. Default: mm to m.")
    parser.add_argument("--no-flip-v", action="store_true")
    parser.add_argument("--max-distance-m", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw_ply).expanduser().resolve()
    texture_path = Path(args.texture).expanduser().resolve()
    geometry_path = Path(args.geometry_mesh).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    raw_vertices, raw_faces, texture_uv, texture_faces = _raw_arrays(raw_path)
    geometry = trimesh.load(geometry_path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = trimesh.util.concatenate(tuple(geometry.geometry.values()))
    if not isinstance(geometry, trimesh.Trimesh):
        raise ValueError(f"Could not load one mesh: {geometry_path}")

    print(f"raw vertices={len(raw_vertices)} faces={len(raw_faces)}", flush=True)
    raw_texture_index = _representative_uv_per_vertex(
        len(raw_vertices), raw_faces, texture_faces,
    )
    tree = cKDTree(np.asarray(raw_vertices, dtype=np.float64) * args.raw_scale)
    distances, nearest = tree.query(np.asarray(geometry.vertices), workers=-1)
    print(
        f"nearest distance m: p50={np.percentile(distances, 50):.7f} "
        f"p95={np.percentile(distances, 95):.7f} max={distances.max():.7f}",
        flush=True,
    )
    if float(distances.max()) > args.max_distance_m:
        raise RuntimeError(
            f"Geometry differs from raw scan: max {distances.max():.6f} m > "
            f"{args.max_distance_m:.6f} m"
        )

    uv = np.asarray(texture_uv[raw_texture_index[nearest]], dtype=np.float32)
    texture = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
    if texture is None:
        raise FileNotFoundError(texture_path)
    height, width = texture.shape[:2]
    map_x = (np.clip(uv[:, 0], 0.0, 1.0) * (width - 1)).astype(np.float32)
    v = uv[:, 1] if args.no_flip_v else 1.0 - uv[:, 1]
    map_y = (np.clip(v, 0.0, 1.0) * (height - 1)).astype(np.float32)
    colors_bgr = cv2.remap(
        texture, map_x.reshape(-1, 1), map_y.reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1, 3)
    colors_rgba = np.column_stack(
        [colors_bgr[:, ::-1], np.full(len(colors_bgr), 255, dtype=np.uint8)]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    coloured = trimesh.Trimesh(
        vertices=np.asarray(geometry.vertices), faces=np.asarray(geometry.faces),
        vertex_normals=np.asarray(geometry.vertex_normals),
        vertex_colors=colors_rgba, process=False,
    )
    coloured.export(output)
    unique = len(np.unique(colors_bgr, axis=0))
    print(
        f"wrote={output} vertices={len(coloured.vertices)} faces={len(coloured.faces)} "
        f"unique_vertex_colors={unique}", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
