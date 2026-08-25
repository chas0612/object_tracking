#!/usr/bin/env python3
"""Executable checks for motion-adaptive sparse stereo scheduling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/process/articulated"))

from make_joint_anchors import (  # noqa: E402
    adaptive_schedule,
    parse_explicit_frames,
    reject_isolated_interpolation_outliers,
)


failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    if condition:
        print(f"  PASS  {label}" + (f"  {detail}" if detail else ""))
    else:
        failures += 1
        print(f"  FAIL  {label}" + (f"  {detail}" if detail else ""))


print("motion-only dense schedule")
anchors = {frame: 0.0 for frame in range(0, 361, 15)}
anchors.update({225: 0.005, 240: 0.080, 255: 0.205, 270: 0.220,
                285: 0.155, 300: 0.010})
frames, windows = adaptive_schedule(
    anchors, set(range(363)), stride=5, movement_threshold=0.020,
    padding_frames=None)
check("drawer motion becomes one padded window", windows == [(210, 315)], str(windows))
check("schedule is dense only inside that window",
      frames == list(range(210, 316, 5)), f"{frames[0]}..{frames[-1]}, n={len(frames)}")
check("stationary beginning and end are not scheduled", 205 not in frames and 320 not in frames)

print("noise does not trigger refinement")
noisy = {0: 0.0, 15: 0.005, 30: 0.010, 45: 0.0}
frames, windows = adaptive_schedule(
    noisy, set(range(46)), stride=5, movement_threshold=0.020,
    padding_frames=None)
check("sub-threshold quantization is ignored", frames == [] and windows == [])

print("explicit targeted schedule")
frames = parse_explicit_frames(["70", "74:86:4", "90:92"], set(range(100)))
check("single frames and inclusive ranges are combined",
      frames == [70, 74, 78, 82, 86, 90, 91, 92], str(frames))
try:
    parse_explicit_frames(["98:102:2"], set(range(100)))
except ValueError as error:
    check("unavailable body poses fail loudly", "unavailable" in str(error), str(error))
else:
    check("unavailable body poses fail loudly", False)

print("isolated angular depth alias")
filtered, rejected = reject_isolated_interpolation_outliers(
    {88: 164.0, 92: 0.0, 96: 162.0, 100: 150.0}, 45.0)
check("a lone zero between consistent open-lid angles is removed",
      rejected == [92] and 92 not in filtered, str((filtered, rejected)))
check("endpoints are always retained", 88 in filtered and 100 in filtered)

print(f"\n{failures} failures")
raise SystemExit(1 if failures else 0)
