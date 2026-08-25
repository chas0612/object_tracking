#!/usr/bin/env python3
"""Pure geometry checks for calibrated center-square cropping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/process"))

from prepare_square_capture import center_square_window, crop_intrinsic  # noqa: E402


failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    if condition:
        print(f"  PASS  {label}")
    else:
        failures += 1
        print(f"  FAIL  {label}  {detail}")


check("portrait crop removes equal vertical borders",
      center_square_window(2000, 2800, 2000) == (0, 400, 2000, 2000))
check("landscape crop removes equal horizontal borders",
      center_square_window(2800, 2000, 2000) == (400, 0, 2000, 2000))

K = [[1000.0, 0.0, 1400.0], [0.0, 1000.0, 900.0], [0.0, 0.0, 1.0]]
cropped = np.asarray(crop_intrinsic(K, 400, 0))
check("crop subtracts its origin from the principal point",
      np.array_equal(cropped, [[1000, 0, 1000], [0, 1000, 900], [0, 0, 1]]), str(cropped))
check("crop does not change focal length", cropped[0, 0] == 1000 and cropped[1, 1] == 1000)

try:
    center_square_window(2000, 2800, 2100)
except ValueError:
    check("oversized crop fails loudly", True)
else:
    check("oversized crop fails loudly", False)

print(f"\n{failures} failures")
raise SystemExit(1 if failures else 0)
