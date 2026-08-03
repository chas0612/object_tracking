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
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOTRACK_DIR = REPO_ROOT / "autodex" / "perception" / "thirdparty" / "MV-GoTrack"
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
    checkpoint = (Path(args.checkpoint).expanduser().resolve() if args.checkpoint else
                  gotrack_dir / "gotrack_checkpoint.pt")
    runner = gotrack_dir / "run_multiview_gotrack_anchor_online_multi_object.py"
    patched_runner = gotrack_dir / "archive" / "run_multiview_gotrack_anchor_online_multi_object.py"
    renderer = gotrack_dir / "utils" / "renderer_nvdiffrast.py"
    for path in (gotrack_dir, runner, patched_runner, renderer):
        if not path.exists():
            errors.append(f"missing required MV-GoTrack path: {path}")
    runner_source = patched_runner.read_text(encoding="utf-8") if patched_runner.is_file() else ""
    if runner_source and "--camera-micro-batch-size" not in runner_source:
        errors.append("MV-GoTrack micro-batch patch is not present in the runner")
    if args.require_articulated:
        # Marker-based, like the micro-batch check above: the articulated patch is
        # applied by hand to a private checkout, so nothing else can tell whether it
        # is there. Without this an articulated run against an unpatched tree does not
        # fail, it silently tracks the object as rigid at whatever angle it was seeded
        # with -- which looks like a plausible result.
        for marker, description in (
            ("--theta-extrapolate-max-deg", "joint-angle prediction"),
            ("--articulation-json", "articulation wiring"),
            # Prismatic support ships in the same patch. An older articulated patch
            # would accept a prismatic joint file and track the object as a hinge at
            # a fixed angle -- the same silent-success failure the checks above guard
            # against, one joint type down.
            ("--joint-extrapolate-max", "prismatic joint support"),
        ):
            if runner_source and marker not in runner_source:
                errors.append(
                    f"MV-GoTrack articulated patch is not present in the runner "
                    f"({description}: {marker} missing)")
        geometry = gotrack_dir / "utils" / "multiview_geometry.py"
        if geometry.is_file():
            geometry_source = geometry.read_text(encoding="utf-8")
            for symbol in ("robust_fit_articulated_pose_from_anchors",
                           "reject_wrong_surface_anchors",
                           "fit_joint_displacement_weighted"):
                if symbol not in geometry_source:
                    errors.append(
                        f"MV-GoTrack articulated patch is incomplete: {symbol} missing "
                        f"from {geometry.name}")
        else:
            errors.append(f"missing required MV-GoTrack path: {geometry}")
    if not checkpoint.is_file():
        errors.append(f"missing GoTrack checkpoint: {checkpoint}")
    elif args.checkpoint_sha256:
        actual = _sha256(checkpoint)
        checks.append(f"checkpoint_sha256={actual}")
        if actual != args.checkpoint_sha256:
            errors.append("GoTrack checkpoint SHA-256 does not match the expected value")

    if str(gotrack_dir) not in sys.path:
        sys.path.insert(0, str(gotrack_dir))
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256,
                        help="Pass an empty string to skip checksum validation.")
    parser.add_argument("--require-repo-sam3", action="store_true",
                        help="Fail if SAM3 is not imported from this repository's thirdparty checkout.")
    parser.add_argument("--require-articulated", action="store_true",
                        help=("Also require patches/MV-GoTrack-articulated.patch to be applied. "
                              "Only the 7-DoF articulated pipeline needs it; 6-DoF runs do not."))
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
