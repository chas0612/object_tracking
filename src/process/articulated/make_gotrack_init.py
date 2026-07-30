#!/usr/bin/env python3
"""Turn one articulated silhouette answer into GoTrack's per-camera init pose files.

The probe already produces exactly what GoTrack needs to start: a body pose and a
lid angle for a single frame, fitted from FoundationPose proposals and chosen by
multi-view silhouettes. GoTrack wants that as one JSON per camera, and it accepts a
world pose directly -- ``extract_pose_camera_from_record`` converts with the
camera's extrinsic -- so nothing has to be re-derived per view.

The lid angle rides along in the same records as ``theta_deg``. The tracker reads
it from there, which keeps the seven degrees of freedom in one file instead of
splitting six into a path and one into a flag that can be forgotten. Forgetting it
is not a small error: a lid seeded closed renders closed templates, the depth
visibility test then rejects every anchor on the open lid, and the angle has no way
back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_cameras
from real_first_pose import DEFAULT_CAPTURE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--frame-index", type=int, default=12,
                        help="Which solved frame to initialise from. The closed frames "
                             "are the safest: they are the ones whose answer is known "
                             "independently, and 12 came back at 0.7 deg against a true 0.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repeat-frames", type=int, default=0,
                        help="Also write the same pose at frames 0..N-1. GoTrack looks the "
                             "init up by frame index and needs one at the frame it starts "
                             "from; this is only a convenience for starting elsewhere.")
    args = parser.parse_args()

    probe = args.capture_dir / "articulated_probe" / f"frame_{args.frame_index:06d}"
    record = json.loads((probe / "hybrid/hybrid_result.json").read_text(encoding="utf-8"))
    answer = record.get("answer") or max(
        (record.get("starts") or record["results"]).values(),
        key=lambda r: r["silhouette_iou"])
    pose_world = np.asarray(answer["pose_body"], dtype=np.float64)
    theta_deg = float(answer["theta_deg"])

    out_dir = args.out or (args.capture_dir / "gotrack_init"
                           / f"frame_{args.frame_index:06d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras(args.capture_dir)
    frames = sorted({args.frame_index, *range(max(0, int(args.repeat_frames)))})
    for camera_id in cameras:
        records = [{
            "frame_index": int(frame),
            "pose_world": pose_world.tolist(),
            "theta_deg": theta_deg,
            "theta_rad": float(np.radians(theta_deg)),
            "source": f"articulated_probe frame {args.frame_index}",
            "silhouette_iou": float(answer["silhouette_iou"]),
            # GoTrack's init fusion scores each view's candidate and drops the ones
            # below --init-min-score, using whichever key --init-weight-key names.
            # These records come from a fit that already pooled every view, so there
            # is no per-view score to report and no per-view ranking to express:
            # every camera carries the same answer. Filling each key with the same
            # positive constant says exactly that -- the views are equally weighted
            # -- rather than leaving the field missing, which reads as a score of
            # zero and makes every candidate fail the threshold.
            **{key: 1.0 for key in (
                "certainty_count_above_threshold",
                "stage3_correspondence_count",
                "inliers_ratio",
                "pose_score",
                "confidence_count_above_threshold",
            )},
        } for frame in frames]
        (out_dir / f"{camera_id}.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8")

    print(f"frame {args.frame_index}: theta {theta_deg:.2f} deg, "
          f"IoU {answer['silhouette_iou']:.4f}", flush=True)
    print(f"wrote {len(cameras)} camera files to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
