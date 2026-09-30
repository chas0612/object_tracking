#!/usr/bin/env python3
"""Preflight checks for one active offline-capture conda environment.

Run once in each environment:

    conda run -n gotrack python scripts/check_offline_capture_setup.py --component gotrack
    conda run -n sam3 python scripts/check_offline_capture_setup.py --component sam3
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOTRACK_DIR = REPO_ROOT / "autodex" / "perception" / "thirdparty" / "MV-GoTrack-next"
DEFAULT_CHECKPOINT_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"

# Several renderer imports transitively import matplotlib. Keep its cache out
# of a potentially read-only home directory on shared compute machines.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/autodex-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_import(name: str, errors: list[str]) -> None:
    try:
        importlib.import_module(name)
    except Exception as exc:  # Report all missing dependencies in one pass.
        errors.append(f"import {name}: {exc!r}")


def _run_sam3(args: argparse.Namespace, errors: list[str], checks: list[str]) -> None:
    for name in ("torch", "torchvision", "cv2", "sam3", "timm", "iopath"):
        _check_import(name, errors)
    try:
        import sam3
        sam3_path = Path(sam3.__file__).resolve()
        expected_root = REPO_ROOT / "autodex" / "perception" / "thirdparty" / "sam3"
        checks.append(f"sam3={sam3_path}")
        if not sam3_path.is_relative_to(expected_root):
            message = f"sam3 source is external to this repo checkout: {sam3_path}"
            if args.require_repo_sam3:
                errors.append(message + f" (expected under {expected_root})")
            else:
                checks.append(message + " (accepted)")
    except Exception:
        pass
    try:
        import torch
        checks.append(f"torch={torch.__version__}")
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is False")
        else:
            checks.append(f"gpu={torch.cuda.get_device_name(0)}")
    except Exception:
        pass


def _run_gotrack(args: argparse.Namespace, errors: list[str], checks: list[str]) -> None:
    gotrack_dir = Path(args.gotrack_dir).expanduser().resolve()
    latest = args.backend == "latest" or (args.backend == "auto" and
                                          (gotrack_dir / "evaluation/dexycb_tracking/adapters.py").is_file())
    checkpoint = (Path(args.checkpoint).expanduser().resolve() if args.checkpoint else
                  gotrack_dir / ("checkpoints/gotrack_checkpoint.pt" if latest else "gotrack_checkpoint.pt"))
    runner = gotrack_dir / "run_multiview_gotrack_anchor_online_multi_object.py"
    patched_runner = gotrack_dir / "archive" / "run_multiview_gotrack_anchor_online_multi_object.py"
    renderer = gotrack_dir / "utils" / "renderer_nvdiffrast.py"
    required = (gotrack_dir, renderer,
                gotrack_dir / "evaluation/dexycb_tracking/adapters.py",
                gotrack_dir / "configs/tracker/irls.json",
                gotrack_dir / "scripts/onboard_custom_mesh_for_foundpose.py") if latest else (
                    gotrack_dir, runner, patched_runner, renderer)
    for path in required:
        if not path.exists():
            errors.append(f"missing required MV-GoTrack path: {path}")
    if not latest and patched_runner.is_file() and "--camera-micro-batch-size" not in patched_runner.read_text(encoding="utf-8"):
        errors.append("MV-GoTrack micro-batch patch is not present in the runner")
    if args.require_articulated:
        target = (gotrack_dir / "evaluation/dexycb_tracking/articulated.py" if latest else patched_runner)
        if not target.is_file():
            errors.append(f"Missing articulated runtime: {target}")
        elif not latest:
            for marker in ("--articulation-json", "--joint-extrapolate-max", "--frame-order", "--joint-anchor-json"):
                if marker not in target.read_text():
                    errors.append(f"Legacy articulated runtime missing {marker}")
    if latest:
        head = subprocess.run(["git", "-C", str(gotrack_dir), "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode or head.stdout.strip() != "842bdfeb61f7f15dd649172077cc72ad6a45d718":
            errors.append("Latest MV-GoTrack must be pinned at 842bdfeb61f7f15dd649172077cc72ad6a45d718")
        if args.require_articulated:
            geometry = gotrack_dir / "utils/multiview_geometry.py"
            source = geometry.read_text() if geometry.is_file() else ""
            for symbol in ("robust_fit_articulated_pose_from_anchors", "reject_wrong_surface_anchors",
                           "fit_joint_displacement_weighted"):
                if symbol not in source:
                    errors.append(f"Articulated patch missing {symbol}")
        for name, sha in (("bop_toolkit", "78940aed90b48209818875b3b2454d2ce88fa835"),
                          ("dinov2", "e1277af2ba9496fbadf7aec6eba56e8d882d1e35")):
            path = gotrack_dir / "external" / name
            result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
            if result.returncode or result.stdout.strip() != sha:
                errors.append(f"{name} must be initialized at {sha}")
    if not checkpoint.is_file():
        errors.append(f"missing GoTrack checkpoint: {checkpoint}")
    elif args.checkpoint_sha256:
        actual = _sha256(checkpoint)
        checks.append(f"checkpoint_sha256={actual}")
        if actual != args.checkpoint_sha256:
            errors.append("GoTrack checkpoint SHA-256 does not match the expected value")

    if str(gotrack_dir) not in sys.path:
        sys.path.insert(0, str(gotrack_dir))
    for dependency in ("bop_toolkit", "dinov2"):
        path = gotrack_dir / "external" / dependency
        sys.path.insert(0, str(path))
    for name in (
        "torch", "torchvision", "cv2", "numpy", "trimesh", "nvdiffrast.torch",
        "hydra", "omegaconf", "pytorch_lightning", "faiss", "kornia", "open3d",
        "pyrender", "OpenGL", "bop_toolkit_lib", "dinov2", "transforms3d", "viser",
    ):
        _check_import(name, errors)
    try:
        import torch
        checks.append(f"torch={torch.__version__}")
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is False")
        else:
            checks.append(f"gpu={torch.cuda.get_device_name(0)}")
            import nvdiffrast.torch as dr
            dr.RasterizeCudaContext()
            checks.append("nvdiffrast_cuda_context=ok")
    except Exception as exc:
        errors.append(f"CUDA/nvdiffrast context: {exc!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("gotrack", "sam3"), required=True)
    parser.add_argument("--gotrack-dir", default=str(DEFAULT_GOTRACK_DIR))
    parser.add_argument("--backend", choices=("auto", "latest", "legacy"), default="auto")
    parser.add_argument("--require-articulated", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256,
                        help="Pass an empty string to skip checksum validation.")
    parser.add_argument("--require-repo-sam3", action="store_true",
                        help="Fail if SAM3 is not imported from this repository's thirdparty checkout.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    args = parser.parse_args()
    errors: list[str] = []
    checks: list[str] = [f"python={sys.executable}", f"component={args.component}"]
    if args.component == "gotrack":
        _run_gotrack(args, errors, checks)
    else:
        _run_sam3(args, errors, checks)
    report = {"ok": not errors, "checks": checks, "errors": errors}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for line in checks:
            print(f"[ok] {line}")
        for line in errors:
            print(f"[error] {line}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
