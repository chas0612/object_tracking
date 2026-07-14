#!/usr/bin/env python3
"""Print progress for ``distribute_foundpose_onboard.py`` without SSH polling.

Example:
    watch -n 5 python scripts/foundpose_onboard_status.py \
      --state-dir ~/shared_data/mesh_blender/.foundpose_onboard_runs/foundpose_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args()
    state_dir = Path(args.state_dir).expanduser().resolve()
    state_path = state_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"State file not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    jobs = state["jobs"]
    counts = Counter(job["status"] for job in jobs.values())
    print(f"run={state.get('run_name')} updated={state.get('updated_utc')}")
    print(" ".join(f"{name}={counts.get(name, 0)}" for name in ("pending", "running", "completed", "failed")))
    for object_name in state.get("objects", sorted(jobs)):
        job = jobs[object_name]
        history = job.get("history", [])
        last = history[-1] if history else {}
        detail = f"attempts={job.get('attempts', 0)}"
        if last:
            detail += f" worker={last.get('worker', '?')} rc={last.get('returncode', 'running')}"
        print(f"{object_name:32} {job['status']:10} {detail}")
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
