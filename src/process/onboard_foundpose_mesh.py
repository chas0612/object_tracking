#!/usr/bin/env python3
"""Build or reuse the per-object FoundPose preprocessing cache on shared storage.

The cache is independent of a particular capture episode.  By default this
script stores it next to the source mesh:

    ~/shared_data/mesh_new/<object>/foundpose_assets/

Run this once per object in the ``gotrack`` environment.  Different PCs can
onboard different objects concurrently because each object has its own cache
and lock.  Two PCs must not onboard the *same* object at the same time.

Example:
    conda run --no-capture-output -n gotrack python -u \
      src/process/onboard_foundpose_mesh.py \
      --object-name apple \
      --reference-intrinsics-json /path/to/capture/cam_param/intrinsics.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GOTRACK_ROOT = REPO_ROOT / "autodex" / "perception" / "thirdparty" / "MV-GoTrack"
ONBOARD_SCRIPT = GOTRACK_ROOT / "scripts" / "onboard_custom_mesh_for_foundpose.py"
DEFAULT_MESH_ROOT = Path.home() / "shared_data" / "mesh_new"


def _validate_object_name(name: str) -> str:
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in name):
        raise ValueError("--object-name may contain only letters, digits, '_', '-', and '.'")
    return name


def _resolve_mesh(object_name: str, mesh_root: Path, explicit_mesh: str | None) -> Path:
    if explicit_mesh:
        mesh = Path(explicit_mesh).expanduser().resolve()
        if not mesh.is_file():
            raise FileNotFoundError(f"Mesh not found: {mesh}")
        return mesh

    object_dir = mesh_root / object_name
    preferred = [object_dir / f"{object_name}{suffix}" for suffix in (".obj", ".ply", ".glb")]
    for mesh in preferred:
        if mesh.is_file():
            return mesh
    candidates = sorted(
        path for suffix in ("*.obj", "*.ply", "*.glb") for path in object_dir.glob(suffix)
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No .obj/.ply/.glb mesh found under {object_dir}. "
            "Pass --mesh explicitly if its filename differs from the object name."
        )
    raise ValueError(f"Multiple meshes found under {object_dir}; pass --mesh explicitly: {candidates}")


def _resolve_intrinsics(args: argparse.Namespace) -> Path:
    if args.reference_intrinsics_json:
        path = Path(args.reference_intrinsics_json).expanduser().resolve()
    else:
        path = Path(args.reference_capture_dir).expanduser().resolve() / "cam_param" / "intrinsics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Reference intrinsics JSON not found: {path}")
    return path


def _resolve_camera_id(intrinsics_path: Path, explicit_id: str | None) -> str:
    payload = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Reference intrinsics JSON is empty or invalid: {intrinsics_path}")
    camera_id = explicit_id or sorted(payload)[0]
    if camera_id not in payload:
        raise KeyError(f"Camera {camera_id} is absent from {intrinsics_path}")
    record = payload[camera_id]
    if not isinstance(record, dict) or "intrinsics_undistort" not in record:
        raise ValueError(f"Camera {camera_id} needs intrinsics_undistort in {intrinsics_path}")
    return camera_id


def _acquire_lock(lock_dir: Path) -> None:
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        owner_path = lock_dir / "owner.json"
        owner = owner_path.read_text(encoding="utf-8") if owner_path.is_file() else "unknown owner"
        raise RuntimeError(
            f"Another onboarding may be running for this object: {lock_dir}\n"
            f"Lock owner: {owner}\n"
            "Do not run the same object on two PCs. After confirming a crashed job, "
            "remove this lock directory manually."
        ) from exc
    owner = {"hostname": socket.gethostname(), "pid": os.getpid(), "started_unix": time.time()}
    (lock_dir / "owner.json").write_text(json.dumps(owner, indent=2) + "\n", encoding="utf-8")


def _release_lock(lock_dir: Path) -> None:
    owner_path = lock_dir / "owner.json"
    if owner_path.exists():
        owner_path.unlink()
    lock_dir.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-name", required=True, help="Object/FoundPose dataset tag.")
    parser.add_argument("--mesh-root", default=str(DEFAULT_MESH_ROOT),
                        help="Default: ~/shared_data/mesh_new")
    parser.add_argument("--mesh", default=None,
                        help="Optional mesh override. Default: <mesh-root>/<object>/<object>.obj")
    parser.add_argument("--output-root", default=None,
                        help="Cache directory. Default: <mesh-root>/<object>/foundpose_assets")
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-intrinsics-json", default=None,
                           help="Calibration JSON whose undistorted K drives template rendering.")
    reference.add_argument("--reference-capture-dir", default=None,
                           help="Uses <capture>/cam_param/intrinsics.json.")
    parser.add_argument("--reference-camera-id", default=None,
                        help="Default: lowest camera ID in the reference intrinsics JSON.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Deprecated compatibility flag. Existing caches are never replaced automatically.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and command only.")
    args = parser.parse_args()

    object_name = _validate_object_name(args.object_name)
    mesh_root = Path(args.mesh_root).expanduser().resolve()
    mesh_path = _resolve_mesh(object_name, mesh_root, args.mesh)
    output_root = (Path(args.output_root).expanduser().resolve() if args.output_root else
                   mesh_root / object_name / "foundpose_assets")
    intrinsics_path = _resolve_intrinsics(args)
    camera_id = _resolve_camera_id(intrinsics_path, args.reference_camera_id)
    repre_path = output_root / "object_repre" / "v1" / object_name / "1" / "repre.pth"
    lock_dir = output_root.parent / f".{object_name}.foundpose_onboard.lock"

    if not ONBOARD_SCRIPT.is_file():
        raise FileNotFoundError(f"FoundPose onboarding script not found: {ONBOARD_SCRIPT}")
    if repre_path.is_file():
        print(f"[skip] completed FoundPose cache: {repre_path}", flush=True)
        return 0
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to modify an existing incomplete cache: {output_root}\n"
            "It is preserved. Choose another output root or inspect it manually."
        )

    # Build outside the final cache path and publish only a complete repre.pth.
    # This makes retries safe after an exception or SSH interruption: a partial
    # onboarding never looks like a reusable cache to FoundPoseInit.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / (
        f".{object_name}.foundpose_assets.staging.{socket.gethostname()}.{os.getpid()}"
    )
    if staging_root.exists():
        raise FileExistsError(f"Unexpected existing staging directory: {staging_root}")

    command = [
        sys.executable, str(ONBOARD_SCRIPT),
        "--mesh-path", str(mesh_path), "--object-id", "1",
        "--dataset-name", object_name, "--output-root", str(staging_root),
        "--reference-intrinsics-json", str(intrinsics_path),
        "--reference-camera-id", camera_id,
        "--reference-image-scale", "1.0", "--mesh-scale", "1000.0",
        "--min-num-viewpoints", "57", "--num-inplane-rotations", "14",
        "--ssaa-factor", "4.0", "--pca-components", "256", "--cluster-num", "2048",
    ]
    print("[foundpose-onboard] object:", object_name, flush=True)
    print("[foundpose-onboard] mesh:", mesh_path, flush=True)
    print("[foundpose-onboard] cache:", output_root, flush=True)
    print("[foundpose-onboard] staging:", staging_root, flush=True)
    print("[foundpose-onboard] reference:", intrinsics_path, "camera", camera_id, flush=True)
    print("[foundpose-onboard] command:", " ".join(command), flush=True)
    if args.dry_run:
        return 0

    _acquire_lock(lock_dir)
    started = time.perf_counter()
    try:
        env = dict(os.environ, PYOPENGL_PLATFORM=os.environ.get("PYOPENGL_PLATFORM", "egl"),
                   EGL_PLATFORM=os.environ.get("EGL_PLATFORM", "surfaceless"))
        subprocess.run(command, check=True, cwd=str(GOTRACK_ROOT), env=env)
        staging_repre = staging_root / "object_repre" / "v1" / object_name / "1" / "repre.pth"
        if not staging_repre.is_file():
            raise RuntimeError(f"Onboarding exited successfully but repre is missing: {staging_repre}")
        manifest = {
            "object_name": object_name,
            "mesh": str(mesh_path),
            "assets_root": str(output_root),
            "staging_root": str(staging_root),
            "reference_intrinsics_json": str(intrinsics_path),
            "reference_camera_id": camera_id,
            "mesh_scale": 1000.0,
            "num_templates_expected": 798,
            "elapsed_sec": time.perf_counter() - started,
            "hostname": socket.gethostname(),
        }
        (staging_root / "preprocess_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging_root.replace(output_root)
        print(f"[done] reusable FoundPose cache: {repre_path}", flush=True)
    finally:
        _release_lock(lock_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
