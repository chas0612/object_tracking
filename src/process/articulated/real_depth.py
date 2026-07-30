#!/usr/bin/env python3
"""Stage 1: stereo depth for the reference camera of each usable pair.

Split out from the FoundationPose stage on purpose. TensorRT stereo needs
``pycuda.autoinit``, which creates its own CUDA context, while FoundationPose
drives NVIDIA Warp kernels. Sharing one process makes Warp resolve the device as
``cuda:0.0`` against arrays on ``cuda:0`` and ``erode_depth_kernel`` refuses to
launch. Writing depth to disk and reading it back in a fresh process sidesteps
that, and matches how the rest of the repo already stages depth.

Depth comes from ``StereoDepthTRT.estimate_pair``, which rectifies, runs the
engine, and un-rectifies with the ``rz`` correction that a naive
``depth = f*B/disparity`` gets wrong -- an error that stays invisible in per-camera
colourmaps and only shows up under cross-view reprojection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
# A TensorRT engine is built for one GPU and one TensorRT version, so it cannot be
# vendored -- it has to be built or copied per machine. Overridable rather than
# hardcoded so a rebuilt engine can be pointed at without editing this file.
STEREO_PLAN = Path(os.environ.get(
    "FOUNDATION_STEREO_PLAN",
    Path.home() / "object-6d-tracking/thirdparty/FoundationStereo"
    / "pretrained_models/foundation_stereo.plan"))

# Measured baselines: 104.6 mm and 74.6 mm. The second pair's right camera is the
# one whose SAM3 mask failed; irrelevant here, since depth needs only pixels and
# the reference (left) camera of each pair does have a mask.
DEFAULT_PAIRS = [("22684755", "23263780"), ("22645026", "22645029")]

from common import load_cameras  # noqa: E402
from real_first_pose import DEFAULT_CAPTURE  # noqa: E402


def _read_frame(video: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {index} from {video}")
        return frame
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--frame-index", type=int, default=40)
    parser.add_argument("--pairs", nargs="*", default=None, help="LEFT:RIGHT specs.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pairs = ([tuple(spec.split(":")) for spec in args.pairs] if args.pairs else DEFAULT_PAIRS)
    frame_dir = args.capture_dir / f"foundpose_frame_{args.frame_index:06d}"
    out_dir = args.out or (args.capture_dir / "articulated_probe"
                           / f"frame_{args.frame_index:06d}" / "depth")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO))
    from autodex.perception.depth import StereoDepthTRT

    cameras = load_cameras(args.capture_dir)
    stereo = StereoDepthTRT(engine_path=str(STEREO_PLAN))
    video_dir = args.capture_dir / "undistorted_video"

    manifest = {}
    for left_id, right_id in pairs:
        if left_id not in cameras or right_id not in cameras:
            print(f"[skip] {left_id}:{right_id} not in calibration", flush=True)
            continue
        left, right = cameras[left_id], cameras[right_id]
        started = time.perf_counter()
        _, depths = stereo.estimate_pair(
            left_rgb=cv2.cvtColor(_read_frame(video_dir / f"{left_id}.avi",
                                              args.frame_index), cv2.COLOR_BGR2RGB),
            right_rgb=cv2.cvtColor(_read_frame(video_dir / f"{right_id}.avi",
                                               args.frame_index), cv2.COLOR_BGR2RGB),
            K_left=left.K, K_right=right.K,
            T_left=left.extrinsic, T_right=right.extrinsic,
            target_serials=[left_id],
            target_intrinsics={left_id: left.K},
            target_extrinsics={left_id: left.extrinsic},
            debug_dir=str(out_dir), pair_label=f"{left_id}_{right_id}")
        if depths is None or left_id not in depths:
            print(f"[skip] {left_id}:{right_id} produced no depth", flush=True)
            continue

        depth = np.nan_to_num(depths[left_id], nan=0.0, posinf=0.0,
                              neginf=0.0).astype(np.float32)
        np.save(out_dir / f"{left_id}_depth.npy", depth)

        mask_path = frame_dir / "masks" / f"{left_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        inside = 0
        if mask is not None:
            mask = mask > 127
            inside = int(((depth > 0) & mask).sum())
            values = depth[(depth > 0) & mask]
            print(f"{left_id}:{right_id}  {time.perf_counter() - started:.1f}s  "
                  f"valid-in-mask={inside} ({100 * inside / max(1, mask.sum()):.1f}%)  "
                  f"range={values.min():.3f}-{values.max():.3f} m  "
                  f"median={np.median(values):.3f} m", flush=True)
        manifest[f"{left_id}:{right_id}"] = {
            "reference_camera": left_id,
            "depth_npy": str(out_dir / f"{left_id}_depth.npy"),
            "valid_in_mask_px": inside,
        }

    (out_dir / "depth_manifest.json").write_text(
        json.dumps({"capture_dir": str(args.capture_dir), "frame_index": args.frame_index,
                    "pairs": manifest}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
