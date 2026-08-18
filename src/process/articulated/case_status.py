#!/usr/bin/env python3
"""Report progress for one run_case.sh articulated tracking attempt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tail", type=int, default=12)
    args = parser.parse_args()

    root = args.capture_dir.expanduser() / "articulated_runs" / args.run_name
    if not root.exists():
        print(f"status=not_started root={root}")
        return 1

    pid_path = root / "pipeline.pid"
    pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
    running = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            running = True
        except (ProcessLookupError, PermissionError):
            pass

    hybrid_files = sorted((root / "probe").glob("frame_*/hybrid/hybrid_result.json"))
    init_count = len(list((root / "gotrack_init").glob("frame_*/*.json")))
    summary_path = root / "gotrack/multi_object_stage_c_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    world_path = root / "gotrack"
    world_records = sorted(world_path.glob("*/world_pose_records.json"))
    tracked = 0
    if world_records:
        try:
            records = json.loads(world_records[0].read_text())
            tracked = len(records) if isinstance(records, (dict, list)) else 0
        except (json.JSONDecodeError, OSError):
            pass

    if (root / "completed").exists():
        status = "completed"
    elif (root / "failed").exists():
        status = "failed"
    elif running:
        status = "running"
    else:
        status = "stopped"
    if status == "completed":
        stage = "done"
    elif not hybrid_files:
        stage = "foundationpose_seed"
    elif init_count < 1:
        stage = "gotrack_init"
    elif not summary:
        stage = "articulated_gotrack"
    else:
        stage = "finalizing"
    print(f"status={status} stage={stage} pid={pid or '-'} root={root}")
    print(f"seed={'done' if hybrid_files else 'pending'} init_json={init_count} "
          f"tracked_records={tracked} summary={'done' if summary else 'pending'}")
    if summary:
        print(f"processed_frames={summary.get('processed_frames')} "
              f"valid_fused_frames={summary.get('valid_fused_frames')} "
              f"runtime_sec={summary.get('runtime_sec')}")

    log_path = root / "logs/pipeline.log"
    if log_path.is_file() and args.tail > 0:
        lines = log_path.read_text(errors="replace").splitlines()
        print(f"--- log tail ({min(args.tail, len(lines))}) ---")
        print("\n".join(lines[-args.tail:]))
    return 0 if status in {"running", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
