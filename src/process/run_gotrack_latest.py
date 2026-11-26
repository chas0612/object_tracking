#!/usr/bin/env python3
"""Capture CLI for the current private MV-GoTrack adapter."""
import argparse
import math
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
    parser.add_argument("--official-config", type=Path,
                        help="Opt-in runner configuration, copied into the run output. Runtime defaults are unchanged.")
    parser.add_argument("--camera-ids", nargs="+", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-anchors", type=int, default=256)
    parser.add_argument("--articulation-json")
    parser.add_argument("--articulated-stability-profile", type=Path,
                        help="Opt-in bounded joint/body fit and explicit body quality gates (JSON).")
    parser.add_argument("--anchor-bank-path")
    parser.add_argument("--body-flow-bootstrap", action="store_true",
                        help="Opt-in real-RGB static-anchor LK/multiview body motion prior.")
    parser.add_argument("--init-joint-value", type=float)
    parser.add_argument("--joint-anchor-json")
    parser.add_argument("--theta-extrapolate-max-deg", type=float)
    parser.add_argument("--joint-extrapolate-max", type=float)
    parser.add_argument("--joint-residual-ratio", type=float,
                        help="Optional moving/static part residual ratio gate; upstream default is unchanged.")
    parser.add_argument("--min-moving-anchors", type=int,
                        help="Optional minimum moving-part observations required to update the joint.")
    parser.add_argument("--max-joint-step-deg", type=float,
                        help="Opt-in revolute measurement gate per frame: reject larger updates and refit body from static anchors.")
    parser.add_argument("--input-resize-scale", type=float, default=0.5)
    parser.add_argument("--first-frame-num-iters", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--observer-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-begin", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--frame-order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--max-frames", type=int, default=-1)
    args = parser.parse_args()
    if args.official_config is not None and not args.official_config.is_file():
        parser.error("--official-config must name an existing JSON file")
    if args.articulation_json and args.backend != "trifit":
        parser.error("Articulated tracking requires --backend trifit")
    if args.body_flow_bootstrap and not args.articulation_json:
        parser.error('--body-flow-bootstrap requires articulation')
    if args.articulated_stability_profile is not None:
        if not args.articulation_json or not args.articulated_stability_profile.is_file():
            parser.error("--articulated-stability-profile requires articulation and an existing JSON file")
    if not args.articulation_json and args.num_anchors not in (128, 256):
        parser.error("Rigid tracking supports 128 or 256 anchors")
    if not args.articulation_json and (args.joint_residual_ratio is not None or args.min_moving_anchors is not None or args.max_joint_step_deg is not None):
        parser.error("Joint observation gates require --articulation-json")
    if args.joint_residual_ratio is not None and (not math.isfinite(args.joint_residual_ratio) or args.joint_residual_ratio <= 0):
        parser.error("--joint-residual-ratio must be finite and positive")
    if args.min_moving_anchors is not None and args.min_moving_anchors < 3:
        parser.error("--min-moving-anchors must be at least 3")
    if args.max_joint_step_deg is not None and (not math.isfinite(args.max_joint_step_deg) or args.max_joint_step_deg <= 0):
        parser.error("--max-joint-step-deg must be finite and positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/autodex-matplotlib")
    from autodex.tracking.latest_gotrack import run_capture
    run_capture(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
