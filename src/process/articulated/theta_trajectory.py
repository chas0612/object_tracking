#!/usr/bin/env python3
"""Read a series of independently-solved frames and ask whether the lid angle moves like a lid.

Everything checked so far is self-consistency: pairs agreeing with pairs,
silhouette agreeing with held-out depth. That is necessary and not sufficient --
the depth objective was self-consistent across three seeds and returned a lid
angle outside the joint's range.

A frame series tests something the single-frame checks cannot. Each frame here was
solved from scratch: its own masks, its own stereo depth, its own registrations,
no seed from its neighbours. Independent estimates of an unrelated quantity would
scatter. Independent estimates of a real lid must land on a smooth curve, because
the lid was on one.

Three readings, in descending order of how much they are worth:

* **The ends.** Frame 36 is closed with the hand already on the lid, and frames
  156-180 have the lid lying flat with the hand gone. The pipeline was told
  neither. Recovering ~0 deg and ~theta_max there is a prediction that can fail.
* **Curvature.** For each interior frame, how far the estimate sits from the line
  through its two neighbours. This is a second difference, so a straight ramp
  scores zero and real curvature scores as curvature -- it bounds the noise from
  above, not exactly.
* **Monotonicity.** The lid is opened in one continuous motion, so the angle
  should never go backwards by more than the estimate's own noise.

The body pose gets the same treatment. It is nearly static -- only nudged by the
hand -- so its translation is a second, independent trajectory on the same frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_articulation
from real_first_pose import DEFAULT_CAPTURE


def _frames(probe_root: Path) -> list[dict]:
    """Every frame under ``probe_root`` that has a finished hybrid result."""
    rows = []
    for path in sorted(probe_root.glob("frame_*/hybrid/hybrid_result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        starts = record.get("starts") or record.get("results") or {}
        if not starts:
            continue
        # The pipeline's own answer where it has one. Older records predate the
        # single-answer output and only have per-pair entries; take the best-scoring
        # of those rather than averaging, so both kinds of record mean the same
        # thing -- one finished fit, not a blend of several.
        answer = record.get("answer") or max(starts.values(),
                                             key=lambda r: r["silhouette_iou"])
        angles = np.array([r["theta_deg"] for r in starts.values()])
        rows.append({
            "frame": record["frame_index"],
            "theta": answer["theta_deg"],
            "theta_spread": record.get("start_spread_deg",
                                       float(angles.max() - angles.min())),
            "selected": record.get("selected_theta_deg"),
            "coarse": sorted({r["coarse_theta_deg"] for r in starts.values()}),
            "iou": answer["silhouette_iou"],
            "centre": np.asarray(answer["pose_body"])[:3, 3],
            "pairs": len(starts),
            "rejected": len(record.get("rejected_pairs") or []),
        })
    return sorted(rows, key=lambda row: row["frame"])


def _second_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """How far each interior point sits from the chord through its neighbours.

    Frame spacing here is deliberately uneven -- tight around the disputed frame,
    loose over the plateau -- so the chord is interpolated by frame index rather
    than by position in the list.
    """
    y = np.asarray(y, dtype=float)
    # Shaped like ``y``, not like ``x``: the angle comes in as a scalar per frame
    # and the body centre as a 3-vector, and both get the same treatment.
    residual = np.full(y.shape, np.nan)
    for i in range(1, len(x) - 1):
        span = x[i + 1] - x[i - 1]
        weight = (x[i] - x[i - 1]) / span
        residual[i] = y[i] - (y[i - 1] * (1 - weight) + y[i + 1] * weight)
    return residual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--object", default="blue_plastic_box")
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    probe_root = args.capture_dir / "articulated_probe"
    rows = _frames(probe_root)
    if len(rows) < 3:
        print(f"only {len(rows)} solved frames under {probe_root}; nothing to trace", flush=True)
        return 1

    theta_max = np.degrees(load_articulation(args.object).theta_max)
    frame = np.array([row["frame"] for row in rows], dtype=float)
    theta = np.array([row["theta"] for row in rows])
    centre = np.stack([row["centre"] for row in rows])

    curvature = _second_difference(frame, theta)
    drift = _second_difference(frame, centre) * 1000.0
    drift = np.linalg.norm(drift, axis=1)

    print(f"{len(rows)} independently solved frames, joint range 0 - {theta_max:.1f} deg\n")
    print("  frame   coarse grid       theta  spread     IoU  pairs  curvature   body")
    for i, row in enumerate(rows):
        grid = ",".join(f"{t:.0f}" for t in row["coarse"])
        bend = "      -" if np.isnan(curvature[i]) else f"{curvature[i]:+7.1f}"
        body = "      -" if np.isnan(drift[i]) else f"{drift[i]:6.1f}m"
        drop = f"  -{row['rejected']}" if row["rejected"] else "   "
        print(f"  {row['frame']:5.0f}   {grid:<14s} {row['theta']:7.1f} {row['theta_spread']:6.1f}"
              f"  {row['iou']:.4f}  {row['pairs']:3d}{drop} {bend} deg {body}m")

    interior = curvature[1:-1]
    steps = np.diff(theta)
    back = steps[steps < 0]
    print(f"\ncurvature: median {np.median(np.abs(interior)):.1f} deg, "
          f"worst {np.max(np.abs(interior)):.1f} deg at frame "
          f"{frame[1:-1][np.argmax(np.abs(interior))]:.0f}")
    print(f"body:      median {np.median(drift[1:-1]):.1f} mm, worst {np.max(drift[1:-1]):.1f} mm")
    print(f"backward:  {len(back)} of {len(steps)} steps go backwards"
          + (f", worst {back.min():.1f} deg" if len(back) else ""))
    print(f"ends:      frame {frame[0]:.0f} -> {theta[0]:.1f} deg (closed is 0), "
          f"frame {frame[-1]:.0f} -> {theta[-1]:.1f} deg (flat open is {theta_max:.1f})")

    # A single number for "did refinement do anything": the coarse stage can only
    # return multiples of the sweep step, so if the refined angles were noise
    # around those multiples the trajectory would be a staircase, not a curve.
    quantised = np.array([row["coarse"][0] for row in rows])
    print(f"refined vs coarse grid: mean |delta| {np.mean(np.abs(theta - quantised)):.1f} deg, "
          f"max {np.max(np.abs(theta - quantised)):.1f} deg")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                             gridspec_kw={"height_ratios": [3, 1]})
        top.axhline(0.0, color="0.8", lw=1)
        top.axhline(theta_max, color="0.8", lw=1)
        top.text(frame[0], theta_max + 4, f"scanned open state, {theta_max:.1f} deg",
                 fontsize=8, color="0.4")
        top.step(frame, quantised, where="mid", color="0.75", lw=1.2,
                 label="coarse 15 deg grid")
        top.plot(frame, theta, "o-", color="#1f77b4", label="refined, per frame")
        for i, row in enumerate(rows):
            if row["theta_spread"] > 0.5:
                top.plot([frame[i], frame[i]],
                         [theta[i] - row["theta_spread"] / 2, theta[i] + row["theta_spread"] / 2],
                         color="#1f77b4", lw=3, alpha=0.4)
        top.set_ylabel("lid angle (deg)")
        top.legend(loc="lower right", fontsize=9)
        top.set_title("Lid angle, each frame solved independently")

        bottom.bar(frame, np.nan_to_num(curvature), width=6, color="#d62728", alpha=0.7)
        bottom.axhline(0, color="0.6", lw=1)
        bottom.set_ylabel("curvature (deg)")
        bottom.set_xlabel("frame")
        figure.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=130)
        print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
