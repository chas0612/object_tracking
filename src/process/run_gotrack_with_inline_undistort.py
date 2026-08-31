#!/usr/bin/env python3
"""Run private MV-GoTrack with an optional top-level inline-undistort adapter."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GOTRACK_ROOT = REPO_ROOT / "autodex/perception/thirdparty/MV-GoTrack"
for path in (REPO_ROOT, GOTRACK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from autodex.tracking.inline_undistort import wrap_capture_dict
from archive import run_multiview_gotrack_anchor_online_multi_object as runner


_original_open_captures = runner._open_captures


def _open_captures(camera_ids, input_root, *, require_masks=True):
    captures = _original_open_captures(
        camera_ids, input_root, require_masks=require_masks,
    )
    return wrap_capture_dict(captures, input_root)


runner._open_captures = _open_captures


if __name__ == "__main__":
    runner.main()
