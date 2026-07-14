#!/usr/bin/env python3
"""Export GoTrack world poses as an animated GLB.

The GLB contains one copy of the object mesh and animation channels for its
translation and rotation.  It is self-contained and can be opened in Blender,
Windows 3D Viewer, or a glTF web viewer.  Source mesh and GoTrack records are
read only.

Example:
    conda run -n gotrack python src/process/export_gotrack_animation_glb.py \
      --mesh /path/to/apple.obj \
      --records /path/to/world_pose_records.json \
      --output /path/to/gotrack_animation.glb
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import trimesh


def _align4(data: bytes, fill: bytes = b"\0") -> bytes:
    return data + fill * ((-len(data)) % 4)


def _append_blob(blob: bytearray, data: bytes) -> tuple[int, int]:
    offset = len(blob)
    blob.extend(_align4(data))
    return offset, len(data)


def _quaternions_xyzw(rotations: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to glTF (x, y, z, w) quaternions."""
    quats = np.empty((len(rotations), 4), dtype=np.float32)
    for i, matrix in enumerate(rotations):
        # Project tiny numerical drift onto SO(3), then use stable trace branches.
        u, _, vt = np.linalg.svd(matrix)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        m00, m01, m02 = rotation[0]
        m10, m11, m12 = rotation[1]
        m20, m21, m22 = rotation[2]
        trace = m00 + m11 + m22
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw, qx, qy, qz = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw, qx, qy, qz = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
        elif m11 > m22:
            s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw, qx, qy, qz = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
        else:
            s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw, qx, qy, qz = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
        quats[i] = (qx, qy, qz, qw)
    # glTF requires consecutive keys to use the same quaternion hemisphere.
    for i in range(1, len(quats)):
        if float(np.dot(quats[i - 1], quats[i])) < 0:
            quats[i] *= -1
    return quats


def _load_poses(records_path: Path) -> tuple[np.ndarray, np.ndarray]:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    by_frame = {
        int(record["frame_index"]): np.asarray(record["pose_world"], dtype=np.float64)
        for record in records if record.get("pose_world") is not None
    }
    if not by_frame:
        raise ValueError(f"No valid pose_world records in {records_path}")
    frames = np.array(sorted(by_frame), dtype=np.int32)
    poses = np.stack([by_frame[int(frame)] for frame in frames])
    if poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("Pose records must contain finite 4x4 matrices")
    return frames, poses


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Could not load a triangle mesh from {mesh_path}")
    return mesh


def export_glb(mesh: trimesh.Trimesh, frames: np.ndarray, poses: np.ndarray, fps: float, output: Path) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape != vertices.shape:
        normals = np.zeros_like(vertices)
    if faces.max() < 65536:
        indices = faces.astype(np.uint16, copy=False).reshape(-1)
        index_component_type = 5123
    else:
        indices = faces.astype(np.uint32, copy=False).reshape(-1)
        index_component_type = 5125

    times = (frames.astype(np.float32) / float(fps)).reshape(-1, 1)
    translations = poses[:, :3, 3].astype(np.float32)
    rotations = _quaternions_xyzw(poses[:, :3, :3])

    binary = bytearray()
    pos_offset, pos_length = _append_blob(binary, vertices.tobytes())
    norm_offset, norm_length = _append_blob(binary, normals.tobytes())
    idx_offset, idx_length = _append_blob(binary, indices.tobytes())
    time_offset, time_length = _append_blob(binary, times.tobytes())
    trans_offset, trans_length = _append_blob(binary, translations.tobytes())
    rot_offset, rot_length = _append_blob(binary, rotations.tobytes())

    buffer_views = [
        {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_length, "target": 34962},
        {"buffer": 0, "byteOffset": norm_offset, "byteLength": norm_length, "target": 34962},
        {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_length, "target": 34963},
        {"buffer": 0, "byteOffset": time_offset, "byteLength": time_length},
        {"buffer": 0, "byteOffset": trans_offset, "byteLength": trans_length},
        {"buffer": 0, "byteOffset": rot_offset, "byteLength": rot_length},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3",
         "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
        {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
        {"bufferView": 2, "componentType": index_component_type, "count": len(indices), "type": "SCALAR"},
        {"bufferView": 3, "componentType": 5126, "count": len(times), "type": "SCALAR",
         "min": [float(times.min())], "max": [float(times.max())]},
        {"bufferView": 4, "componentType": 5126, "count": len(translations), "type": "VEC3"},
        {"bufferView": 5, "componentType": 5126, "count": len(rotations), "type": "VEC4"},
    ]
    document = {
        "asset": {"version": "2.0", "generator": "AutoDex GoTrack animation exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "apple_gotrack_pose", "mesh": 0}],
        "meshes": [{"name": "tracked_object", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0,
        }]}],
        "materials": [{"name": "tracked_object_material", "pbrMetallicRoughness": {
            "baseColorFactor": [0.75, 0.2, 0.1, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.65,
        }}],
        "animations": [{"name": "gotrack_6d_pose", "samplers": [
            {"input": 3, "output": 4, "interpolation": "LINEAR"},
            {"input": 3, "output": 5, "interpolation": "LINEAR"},
        ], "channels": [
            {"sampler": 0, "target": {"node": 0, "path": "translation"}},
            {"sampler": 1, "target": {"node": 0, "path": "rotation"}},
        ]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_chunk = _align4(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    binary_chunk = _align4(bytes(binary))
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        handle.write(json_chunk)
        handle.write(struct.pack("<I4s", len(binary_chunk), b"BIN\0"))
        handle.write(binary_chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--records", required=True, help="GoTrack world_pose_records.json")
    parser.add_argument("--output", required=True, help="Output .glb path")
    parser.add_argument("--fps", type=float, default=30.0, help="Animation rate matching the capture videos")
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    mesh = _load_mesh(Path(args.mesh).expanduser().resolve())
    frames, poses = _load_poses(Path(args.records).expanduser().resolve())
    export_glb(mesh, frames, poses, args.fps, output)
    print(f"[done] {output} ({len(frames)} animation frames, {frames[-1] / args.fps:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
