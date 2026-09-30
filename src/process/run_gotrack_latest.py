#!/usr/bin/env python3
"""Capture CLI for the current private MV-GoTrack adapter."""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gotrack-dir", required=True)
    parser.add_argument("--backend", choices=("irls", "trifit"), required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mesh-path", required=True)
    parser.add_argument("--init-pose-source", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--camera-ids", nargs="+", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-anchors", type=int, default=256)
    parser.add_argument("--articulation-json")
    parser.add_argument("--anchor-bank-path")
    parser.add_argument("--init-joint-value", type=float)
    parser.add_argument("--joint-anchor-json")
    parser.add_argument("--theta-extrapolate-max-deg", type=float)
    parser.add_argument("--joint-extrapolate-max", type=float)
    parser.add_argument("--input-resize-scale", type=float, default=0.5)
    parser.add_argument("--first-frame-num-iters", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--observer-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-begin", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--frame-order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--max-frames", type=int, default=-1)
    args = parser.parse_args()
    if args.articulation_json and args.backend != "trifit":
        parser.error("Articulated tracking requires --backend trifit")
    if not args.articulation_json and args.num_anchors not in (128, 256):
        parser.error("Rigid tracking supports 128 or 256 anchors")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/autodex-matplotlib")
    from autodex.tracking.latest_gotrack import run_capture
    run_capture(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
