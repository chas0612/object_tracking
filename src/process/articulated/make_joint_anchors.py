#!/usr/bin/env python3
"""Measure one articulated joint from stereo depth at selected frames.

Why this exists. Optical flow cannot observe a slide that runs along the viewing rays,
and the failure is not noisy but *coherent*: every anchor on a sliding part moves by
the same vector, so a drift of the whole part is arithmetically identical to a real
displacement and no residual test can separate them. Measured on drawer/2 frame 292,
thirty-seven drawer anchors agreed to within 20 mm on a value 80 mm wrong while the
residual ratio read a healthy 1.23. Two of the three drawer captures then walked the
joint into its ceiling and lost the object outright.

Depth does see it, because it measures absolute position rather than change: at that
frame the truth scores 0.553 against 0.441 for the tracker's answer.

**The body pose is not corrected and does not need to be.** Sweeping the body along the
slide axis and re-scoring shows its depth agreement peaking within a millimetre of
where tracking put it, while the joint's optimum slides one-for-one with it -- the two
are coupled, and depth pins the body but not the joint. So this reads the body pose
from an existing run and solves the one degree of freedom that flow got wrong.

That is also why this is a second pass rather than something inside the tracker: the
joint measurement needs a body pose, the body pose comes from tracking, and tracking
is what we are correcting.  The measurements can either pin a second tracker run or
feed ``constrain_joint_trajectory.py`` without rerunning GoTrack.

For a revolute joint this is also a recovery path for short, violent body motions:
track the parent rigidly, hold that body pose fixed, and sweep only the moving part's
angle at named frames.  It avoids asking optical flow to separate a simultaneous body
rotation from a hinge rotation.  Values written for a hinge are radians, exactly like
``world_pose_records.json``; display and CLI sweep spacing may use degrees.

Anchoring every frame is possible and costs about 40 minutes for a 363-frame capture.
The default spacing is a compromise.  A second adaptive pass can reuse that coarse
JSON, detect intervals whose two stereo measurements differ, and measure only those
intervals more densely.  This motion detector intentionally uses absolute stereo
rather than GoTrack: the latter's coherent drift is the reason anchors are needed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import load_articulation
from depth_joint import DepthJointObjective

DEFAULT_PAIRS = ["22684755:23263780", "22645021:23180202",
                 "25452066:26256735", "22645026:22645029"]


def parse_explicit_frames(specs: list[str], available_frames: set[int]) -> list[int]:
    """Parse individual frames and inclusive ``START:END[:STEP]`` ranges."""
    selected: set[int] = set()
    for spec in specs:
        fields = spec.split(":")
        if len(fields) == 1:
            selected.add(int(fields[0]))
            continue
        if len(fields) not in (2, 3):
            raise ValueError(f"invalid --frames item {spec!r}; use N or START:END[:STEP]")
        start, end = int(fields[0]), int(fields[1])
        step = int(fields[2]) if len(fields) == 3 else 1
        if step <= 0 or end < start:
            raise ValueError(f"invalid --frames range {spec!r}")
        selected.update(range(start, end + 1, step))
    missing = sorted(selected - available_frames)
    if missing:
        raise ValueError(f"--frames contains unavailable body-pose frames: {missing}")
    return sorted(selected)


def reject_isolated_interpolation_outliers(
    anchors: dict[int, float], max_error: float,
) -> tuple[dict[int, float], list[int]]:
    """Drop isolated scalar aliases far from interpolation of their neighbours.

    Endpoints are never removed.  This is opt-in because a genuinely impulsive joint
    motion can also look isolated at a sparse sampling rate; it is intended for an
    obvious depth alias such as ``164, 0, 162`` degrees on consecutive samples.
    """
    if max_error <= 0.0:
        raise ValueError("maximum interpolation error must be positive")
    ordered = sorted(anchors)
    rejected = []
    kept = dict(anchors)
    for left, frame, right in zip(ordered, ordered[1:], ordered[2:]):
        alpha = (frame - left) / float(right - left)
        expected = anchors[left] + alpha * (anchors[right] - anchors[left])
        # An isolated alias has mutually consistent neighbours.  Requiring that
        # agreement prevents a bad sample from making its next (valid) neighbour
        # look like another outlier, e.g. 164, 0, 162, 150 degrees.
        neighbours_agree = abs(anchors[right] - anchors[left]) <= max_error
        if neighbours_agree and abs(anchors[frame] - expected) > max_error:
            rejected.append(frame)
            kept.pop(frame, None)
    return kept, rejected


def load_existing_anchors(path: Path, object_name: str) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload.get(object_name)
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{path}: no non-empty anchor table for {object_name!r}")
    anchors = {int(frame): float(value) for frame, value in table.items()}
    if not all(np.isfinite(value) for value in anchors.values()):
        raise ValueError(f"{path}: anchor values must be finite")
    return anchors


def adaptive_schedule(
    anchors: dict[int, float],
    available_frames: set[int],
    *,
    stride: int,
    movement_threshold: float,
    padding_frames: int | None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return dense candidates around coarse intervals that contain motion.

    One neighbouring coarse interval is included on each side by default.  It covers
    acceleration before the first large displacement and deceleration after the last
    one, and also bridges a near-constant pair at the turn-around of a motion.
    """
    if stride <= 0:
        raise ValueError("adaptive stride must be positive")
    if movement_threshold <= 0.0:
        raise ValueError("adaptive movement threshold must be positive")
    ordered = sorted(anchors)
    if len(ordered) < 2:
        raise ValueError("adaptive refinement needs at least two coarse anchors")
    gaps = np.diff(ordered)
    padding = int(np.median(gaps)) if padding_frames is None else int(padding_frames)
    if padding < 0:
        raise ValueError("adaptive padding must be non-negative")

    intervals = []
    for left, right in zip(ordered, ordered[1:]):
        if abs(anchors[right] - anchors[left]) >= movement_threshold:
            intervals.append((left - padding, right + padding))
    if not intervals:
        return [], []

    # Merge overlap before scheduling so both the log and the cost estimate describe
    # the actual motion windows, not a long list of pairwise detections.
    merged: list[list[int]] = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    windows = [(max(min(available_frames), left), min(max(available_frames), right))
               for left, right in merged]
    frames = sorted(frame for frame in available_frames
                    if frame % stride == 0
                    and any(left <= frame <= right for left, right in windows))
    return frames, windows


def ensure_depth(capture_dir: Path, frame: int, pairs: list[str]) -> Path:
    """Stereo depth for one frame, computed only if it is not already there.

    No mask is needed or used: ``real_depth.py`` reads one only to report how much of
    its output landed inside it, and the objective here scores the moving part's own
    pixels wherever they fall. That matters for cost -- segmentation is the expensive
    step and this skips it entirely.
    """
    manifest = capture_dir / "articulated_probe" / f"frame_{frame:06d}" / "depth" / "depth_manifest.json"
    if manifest.exists():
        return manifest
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("real_depth.py")),
         "--capture-dir", str(capture_dir), "--frame-index", str(frame),
         "--pairs", *pairs],
        check=True, cwd=str(Path(__file__).parent),
        stdout=subprocess.DEVNULL)
    if not manifest.exists():
        raise RuntimeError(f"frame {frame}: real_depth.py produced no {manifest}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--pose-object", default=None,
                        help="Object directory holding body pose records. Defaults to "
                             "--object; use body_1F for a rigid parent bootstrap.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A completed run whose body poses to read, i.e. the "
                             "<run>/gotrack directory of a previous pass.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=15,
                        help="Anchor every Nth frame. Smaller costs proportionally "
                             "more and leaves less room for the joint to drift inside "
                             "a gap; see the module docstring.")
    parser.add_argument("--frames", nargs="*", default=None,
                        help="Explicit frames, as N or inclusive START:END[:STEP]. "
                             "Overrides --stride/adaptive scheduling; useful for a "
                             "known short failure interval.")
    parser.add_argument("--adaptive-from", type=Path, default=None,
                        help="Existing coarse anchor JSON. Detect moving intervals "
                             "from its stereo values, preserve values outside them, "
                             "and re-score every dense frame inside them against the "
                             "current run's body pose. Existing depth is reused.")
    parser.add_argument("--adaptive-stride", type=int, default=5,
                        help="Frame spacing inside detected moving intervals.")
    parser.add_argument("--adaptive-movement-threshold", type=float, default=0.02,
                        help="Minimum displacement between adjacent coarse anchors "
                             "that marks motion, in the joint's native unit (default "
                             "0.02 m). Keep this above coarse stereo quantization.")
    parser.add_argument("--adaptive-padding-frames", type=int, default=None,
                        help="Pad each detected interval on both sides. Default: one "
                             "median coarse-anchor gap.")
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    parser.add_argument("--scale", type=float, default=0.15,
                        help="Resolution the agreement is scored at. 0.15 answered "
                             "within 5 mm of 0.25 on drawer/2 at a third of the cost.")
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--joint-min", type=float, default=None,
                        help="Lower search bound in native units (radians or metres).")
    parser.add_argument("--joint-max", type=float, default=None,
                        help="Upper search bound in native units (radians or metres).")
    parser.add_argument("--step", type=float, default=None,
                        help="Native-unit sweep spacing. Default: 0.005 m for a slide "
                             "or 2 degrees for a hinge.")
    parser.add_argument("--step-deg", type=float, default=None,
                        help="Hinge sweep spacing in degrees. Revolute only; mutually "
                             "exclusive with --step.")
    parser.add_argument("--min-contrast", type=float, default=0.06,
                        help="Skip a frame whose agreement curve is this flat, peak "
                             "above median. A shut drawer is nearly unobservable even "
                             "to depth -- its own surface barely moves in any view -- "
                             "and anchoring to the argmax of a flat curve would pin "
                             "the joint to noise.")
    parser.add_argument("--max-interpolation-error-deg", type=float, default=None,
                        help="Revolute-only, opt-in removal of an isolated anchor this "
                             "far from interpolation of its two neighbours. Endpoints "
                             "are retained.")
    args = parser.parse_args()

    articulation = load_articulation(args.object)
    if args.step is not None and args.step_deg is not None:
        raise ValueError("pass --step or --step-deg, not both")
    if args.step_deg is not None and articulation.joint_type != "revolute":
        raise ValueError("--step-deg is only defined for a revolute joint")

    pose_object = args.object if args.pose_object is None else args.pose_object
    records = json.loads(
        (args.run_dir / pose_object / "world_pose_records.json").read_text(encoding="utf-8"))
    poses = {int(r["frame_index"]): np.asarray(r["pose_world"], dtype=np.float64)
             for r in records if r.get("status") == "ok"}
    if not poses:
        raise ValueError(f"{args.run_dir}: no solved frames to read body poses from")

    intrinsics = json.loads(
        (args.capture_dir / "cam_param/intrinsics.json").read_text(encoding="utf-8"))
    extrinsics = json.loads(
        (args.capture_dir / "cam_param/extrinsics.json").read_text(encoding="utf-8"))

    lower = articulation.joint_min if args.joint_min is None else float(args.joint_min)
    upper = articulation.joint_max if args.joint_max is None else float(args.joint_max)
    if lower > upper:
        raise ValueError("--joint-min exceeds --joint-max")
    if args.step_deg is not None:
        step = float(np.radians(args.step_deg))
    elif args.step is not None:
        step = float(args.step)
    else:
        step = float(np.radians(2.0) if articulation.joint_type == "revolute" else 0.005)
    if step <= 0.0:
        raise ValueError("joint sweep step must be positive")
    grid = np.arange(lower, upper + 1e-12, step)
    if grid.size == 0 or grid[-1] < upper - 1e-12:
        grid = np.append(grid, upper)

    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.frames is not None:
        anchors = ({ } if args.adaptive_from is None else
                   load_existing_anchors(args.adaptive_from, args.object))
        frames = parse_explicit_frames(args.frames, set(poses))
        schedule = f"{len(frames)} explicitly selected frames"
    elif args.adaptive_from is None:
        anchors: dict[int, float] = {}
        frames = [f for f in range(0, max(poses) + 1, args.stride) if f in poses]
        schedule = f"{len(frames)} candidate frames at stride {args.stride}"
    else:
        anchors = load_existing_anchors(args.adaptive_from, args.object)
        frames, windows = adaptive_schedule(
            anchors, set(poses), stride=args.adaptive_stride,
            movement_threshold=args.adaptive_movement_threshold,
            padding_frames=args.adaptive_padding_frames)
        reused_depth = sum(
            (args.capture_dir / "articulated_probe" / f"frame_{frame:06d}" /
             "depth/depth_manifest.json").is_file()
            for frame in frames)
        window_text = ", ".join(f"{left}..{right}" for left, right in windows) or "none"
        schedule = (f"{len(frames)} refinement frames at adaptive stride "
                    f"{args.adaptive_stride} ({reused_depth} reuse existing depth); "
                    f"motion windows {window_text}; preserving coarse values outside")
    print(f"{args.object}: {schedule}, searching "
          f"{articulation.display(grid[0]):.0f}..{articulation.display(grid[-1]):.0f} "
          f"{articulation.joint_unit}", flush=True)

    skipped = []
    for frame in frames:
        manifest = ensure_depth(args.capture_dir, frame, list(args.pairs))
        objective = DepthJointObjective(articulation, manifest, intrinsics, extrinsics,
                                        scale=args.scale, samples=args.samples)
        values = np.array([objective.agreement(poses[frame], g) for g in grid])
        best = int(np.argmax(values))
        contrast = float(values[best] - np.median(values))
        if contrast < args.min_contrast:
            skipped.append(frame)
            print(f"  frame {frame:5d}: flat curve (contrast {contrast:.3f}), not anchored",
                  flush=True)
            continue
        anchors[frame] = float(grid[best])
        print(f"  frame {frame:5d}: {articulation.display(grid[best]):7.1f} "
              f"{articulation.joint_unit}  agreement {values[best]:.3f}  "
              f"contrast {contrast:.3f}", flush=True)

    alias_rejected: list[int] = []
    if args.max_interpolation_error_deg is not None:
        if articulation.joint_type != "revolute":
            raise ValueError("--max-interpolation-error-deg requires a revolute joint")
        anchors, alias_rejected = reject_isolated_interpolation_outliers(
            anchors, float(np.radians(args.max_interpolation_error_deg)))
        if alias_rejected:
            print(f"  rejected isolated angular aliases at frames {alias_rejected}",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({args.object: anchors}, indent=1) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {len(anchors)} anchors to {args.out}"
          + (f" ({len(skipped)} frames skipped as unobservable)" if skipped else ""),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
