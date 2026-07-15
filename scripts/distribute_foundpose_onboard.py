#!/usr/bin/env python3
"""Distribute restart-safe FoundPose mesh onboarding across SSH workers.

Run this controller inside a local tmux session. It gives every worker at most
one object at a time, continues after individual SSH/onboarding failures, and
persists state plus per-attempt logs under the shared mesh root. Re-running the
same command resumes unfinished objects and skips completed representations.

Example (replace IPs with the verified addresses):

    tmux new -s foundpose-onboard
    python -u scripts/distribute_foundpose_onboard.py \
      --all-mesh-objects \
      --workers local capture13@192.168.0.113 capture14@192.168.0.114 \
                capture15@192.168.0.115 capture18@192.168.0.118

Workers are specified as ``local`` or ``user@host``. The remote repository is
assumed at ``$HOME/object_tracking`` and shared storage at ``$HOME/shared_data``;
override their relative paths when needed.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Worker:
    name: str
    ssh_target: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_worker(spec: str) -> Worker:
    if spec == "local":
        return Worker(name="local", ssh_target=None)
    if "@" not in spec or spec.endswith("@"):
        raise ValueError(f"Worker must be 'local' or user@host, got: {spec}")
    return Worker(name=spec.replace("@", "_at_").replace("/", "_"), ssh_target=spec)


def _load_objects(args: argparse.Namespace, mesh_root: Path) -> list[str]:
    objects = list(args.objects)
    if args.objects_file:
        for line in Path(args.objects_file).expanduser().read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                objects.append(line)
    if args.all_mesh_objects:
        objects.extend(
            directory.name
            for directory in sorted(mesh_root.iterdir())
            if directory.is_dir() and not directory.name.startswith(".") and any(
                path.is_file() for suffix in ("*.obj", "*.ply", "*.glb")
                for path in directory.glob(suffix)
            )
        )
    unique = list(dict.fromkeys(objects))
    if not unique:
        raise ValueError("Pass --objects, --objects-file, and/or --all-mesh-objects")
    invalid = [obj for obj in unique if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in obj)]
    if invalid:
        raise ValueError(f"Invalid object names: {invalid}")
    return unique


def _reference_intrinsics_by_object(
    *,
    objects: list[str],
    shared_root: Path,
    fallback_rel: str,
    auto_reference: bool,
) -> dict[str, str]:
    """Pick each object's own capture calibration, falling back deterministically.

    A matching capture is one whose path beneath ``shared_data/capture`` has an
    exact directory component equal to the mesh object name.  This deliberately
    avoids fuzzy filename matching (for example, ``pan`` matching an unrelated
    path).  Only paths that actually contain ``cam_param/intrinsics.json`` are
    candidates.  The first stable candidate is enough because FoundPose
    onboarding only needs a representative undistorted camera model.
    """
    fallback = shared_root / fallback_rel
    if not fallback.is_file():
        raise FileNotFoundError(f"Fallback reference intrinsics is missing: {fallback}")
    capture_root = shared_root / "capture"
    all_intrinsics = (
        sorted(capture_root.rglob("intrinsics.json")) if auto_reference and capture_root.is_dir() else []
    )
    resolved: dict[str, str] = {}
    for object_name in objects:
        candidates = [
            path for path in all_intrinsics
            if object_name in path.relative_to(capture_root).parts
        ]
        selected = min(candidates, key=lambda path: (len(path.parts), str(path))) if candidates else fallback
        resolved[object_name] = str(selected.relative_to(shared_root))
    return resolved


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _completed(output_root: Path, object_name: str) -> bool:
    return (output_root / "object_repre" / "v1" /
            object_name / "1" / "repre.pth").is_file()


def _has_direct_mesh(mesh_root: Path, object_name: str) -> bool:
    object_dir = mesh_root / object_name
    return object_dir.is_dir() and any(
        path.is_file() for suffix in ("*.obj", "*.ply", "*.glb")
        for path in object_dir.glob(suffix)
    )


def _scenario_object_paths(
    *, shared_root: Path, mesh_root: Path, scenario_root_rel: str
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Use episode-0 calibration and a campaign-local cache for each object."""
    scenario_root = shared_root / scenario_root_rel
    if not scenario_root.is_dir():
        raise FileNotFoundError(f"Scenario root is missing: {scenario_root}")
    objects: list[str] = []
    references: dict[str, str] = {}
    outputs: dict[str, str] = {}
    for object_dir in sorted(scenario_root.iterdir()):
        if not object_dir.is_dir() or object_dir.name.startswith("."):
            continue
        object_name = object_dir.name
        reference = object_dir / "0" / "cam_param" / "intrinsics.json"
        if not reference.is_file() or not _has_direct_mesh(mesh_root, object_name):
            continue
        objects.append(object_name)
        references[object_name] = str(reference.relative_to(shared_root))
        outputs[object_name] = str((object_dir / "foundpose_assets").relative_to(shared_root))
    if not objects:
        raise ValueError(f"No scenario objects with episode-0 calibration and a mesh: {scenario_root}")
    return objects, references, outputs


def _run_one(
    worker: Worker,
    object_name: str,
    attempt: int,
    reference_intrinsics_rel: str,
    output_root_rel: str,
    args: argparse.Namespace,
    log_path: Path,
) -> tuple[int, str]:
    shared_root = Path.home() / args.shared_root_rel
    mesh_root = shared_root / "mesh_blender"
    reference_json = shared_root / reference_intrinsics_rel
    output_root = shared_root / output_root_rel
    local_command = [
        "conda", "run", "--no-capture-output", "-n", args.gotrack_env, "python", "-u",
        "src/process/onboard_foundpose_mesh.py", "--object-name", object_name,
        "--mesh-root", str(mesh_root), "--output-root", str(output_root),
        "--reference-intrinsics-json", str(reference_json),
    ]
    if worker.ssh_target is None:
        command = local_command
        cwd = str(REPO_ROOT)
    else:
        remote_repo = f'$HOME/{args.remote_repo_rel.strip("/")}'
        remote_shared = f'$HOME/{args.shared_root_rel.strip("/")}'
        remote_command = [
            "$HOME/anaconda3/bin/conda", "run", "--no-capture-output", "-n", args.gotrack_env, "python", "-u",
            "src/process/onboard_foundpose_mesh.py", "--object-name", object_name,
            "--mesh-root", f"{remote_shared}/mesh_blender",
            "--output-root", f"{remote_shared}/{output_root_rel.lstrip('/')}",
            "--reference-intrinsics-json", f"{remote_shared}/{reference_intrinsics_rel.lstrip('/')}",
        ]
        # Quote each argument for the remote shell, preserving $HOME expansion.
        rendered = " ".join(
            part if part.startswith("$HOME/") else shlex.quote(part) for part in remote_command
        )
        remote_script = f"set -euo pipefail; cd {remote_repo}; exec {rendered}"
        command = ["ssh", "-p", "77", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={args.connect_timeout}",
                   worker.ssh_target, remote_script]
        cwd = None

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_utc={_utc_now()} worker={worker.name} object={object_name} attempt={attempt}\n")
        log.write("command=" + " ".join(shlex.quote(part) for part in command) + "\n\n")
        log.flush()
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\nfinished_utc={_utc_now()} returncode={result.returncode}\n")
    return result.returncode, str(log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", nargs="*", default=[])
    parser.add_argument("--objects-file", default=None, help="One object per line; # starts comments.")
    parser.add_argument("--all-mesh-objects", action="store_true",
                        help="Onboard every immediate mesh_blender subdirectory with an .obj/.ply/.glb mesh.")
    parser.add_argument("--scenario-root-rel", default=None,
                        help="Restrict to a campaign root under shared_data. Each <object>/0 calibration is "
                             "used and assets are saved as <object>/foundpose_assets.")
    parser.add_argument("--workers", nargs="+", required=True,
                        help="local and/or user@ip. One object runs per worker.")
    parser.add_argument("--fallback-reference-intrinsics-rel",
                        default="capture/eccv2026/inspire_dftp/apple/0/cam_param/intrinsics.json",
                        help="Fallback relative to ~/shared_data when no object-specific capture exists.")
    parser.add_argument("--no-auto-reference-intrinsics", dest="auto_reference_intrinsics",
                        action="store_false",
                        help="Always use --fallback-reference-intrinsics-rel; do not search captures by object name.")
    parser.set_defaults(auto_reference_intrinsics=True)
    parser.add_argument("--shared-root-rel", default="shared_data",
                        help="Shared storage relative to each worker HOME. Default: shared_data")
    parser.add_argument("--remote-repo-rel", default="object_tracking",
                        help="Repository relative to remote HOME. Default: object_tracking")
    parser.add_argument("--gotrack-env", default="gotrack")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Attempts per object before recording it failed. Default: 3")
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--run-name", default=None,
                        help="Resume this shared state directory. Default: timestamped new run.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="On resume, reset terminal failed objects to pending.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_attempts < 1 or args.connect_timeout < 1:
        raise ValueError("--max-attempts and --connect-timeout must be positive")
    if args.fallback_reference_intrinsics_rel.startswith("/") or ".." in Path(args.fallback_reference_intrinsics_rel).parts:
        raise ValueError("--fallback-reference-intrinsics-rel must be a safe path relative to shared_data")
    if args.scenario_root_rel and (args.scenario_root_rel.startswith("/") or ".." in Path(args.scenario_root_rel).parts):
        raise ValueError("--scenario-root-rel must be a safe path relative to shared_data")

    shared_root = Path.home() / args.shared_root_rel
    mesh_root = shared_root / "mesh_blender"
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"Mesh root is missing: {mesh_root}")
    if args.scenario_root_rel:
        if args.objects or args.objects_file or args.all_mesh_objects:
            raise ValueError("--scenario-root-rel cannot be combined with --objects, --objects-file, or --all-mesh-objects")
        objects, reference_by_object, output_by_object = _scenario_object_paths(
            shared_root=shared_root, mesh_root=mesh_root, scenario_root_rel=args.scenario_root_rel,
        )
    else:
        objects = _load_objects(args, mesh_root)
        reference_by_object = _reference_intrinsics_by_object(
            objects=objects,
            shared_root=shared_root,
            fallback_rel=args.fallback_reference_intrinsics_rel,
            auto_reference=args.auto_reference_intrinsics,
        )
        output_by_object = {
            obj: str((mesh_root / obj / "foundpose_assets").relative_to(shared_root)) for obj in objects
        }
    workers = [_parse_worker(spec) for spec in args.workers]
    if len({worker.name for worker in workers}) != len(workers):
        raise ValueError("Worker names must be unique")
    run_name = args.run_name or datetime.now().strftime("foundpose_%Y%m%d_%H%M%S")
    state_dir = mesh_root / ".foundpose_onboard_runs" / run_name
    state_path = state_dir / "state.json"
    logs_dir = state_dir / "logs"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        jobs = state["jobs"]
        unknown = sorted(set(objects) - set(jobs))
        if unknown:
            raise ValueError(f"Objects absent from existing run {run_name}: {unknown}")
    else:
        state_dir.mkdir(parents=True, exist_ok=False)
        logs_dir.mkdir()
        jobs = {
            obj: {"status": "pending", "attempts": 0, "history": [],
                  "reference_intrinsics_rel": reference_by_object[obj],
                  "output_root_rel": output_by_object[obj]}
            for obj in objects
        }
        state = {"run_name": run_name, "created_utc": _utc_now(), "objects": objects, "jobs": jobs}

    # A resumed old run may predate per-object reference selection. Preserve an
    # already-recorded reference for reproducibility; add it only when absent.
    for obj in objects:
        jobs[obj].setdefault("reference_intrinsics_rel", reference_by_object[obj])
        jobs[obj].setdefault("output_root_rel", output_by_object[obj])

    for obj in objects:
        if _completed(shared_root / str(jobs[obj]["output_root_rel"]), obj):
            jobs[obj]["status"] = "completed"
        elif jobs[obj]["status"] == "failed" and args.retry_failed:
            jobs[obj]["status"] = "pending"
    state["updated_utc"] = _utc_now()
    _atomic_json(state_path, state)

    print(f"[scheduler] state={state_dir}")
    print(f"[scheduler] workers={' '.join(worker.name for worker in workers)}")
    for obj in objects:
        source = jobs[obj]["reference_intrinsics_rel"]
        label = "fallback" if source == args.fallback_reference_intrinsics_rel else "object-capture"
        print(f"[reference:{label}] {obj} <- {source}; cache={jobs[obj]['output_root_rel']}")
    if args.dry_run:
        for worker in workers:
            for obj in objects:
                log_path = logs_dir / f"{obj}.attempt1.{worker.name}.log"
                print(f"[dry-run] {worker.name} <- {obj}: {log_path}")
        return 0

    pending = deque(obj for obj in objects if jobs[obj]["status"] == "pending")
    active: dict[Future[tuple[int, str]], tuple[Worker, str, int]] = {}
    available = deque(workers)
    state_lock = Lock()
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        while pending or active:
            while pending and available:
                worker = available.popleft()
                obj = pending.popleft()
                if _completed(shared_root / str(jobs[obj]["output_root_rel"]), obj):
                    jobs[obj]["status"] = "completed"
                    available.append(worker)
                    continue
                attempt = int(jobs[obj]["attempts"]) + 1
                jobs[obj]["status"] = "running"
                jobs[obj]["attempts"] = attempt
                jobs[obj]["history"].append({"attempt": attempt, "worker": worker.name, "started_utc": _utc_now()})
                _atomic_json(state_path, state)
                log_path = logs_dir / f"{obj}.attempt{attempt}.{worker.name}.log"
                future = executor.submit(
                    _run_one, worker, obj, attempt,
                    str(jobs[obj]["reference_intrinsics_rel"]),
                    str(jobs[obj]["output_root_rel"]), args, log_path,
                )
                active[future] = (worker, obj, attempt)
                print(f"[start] {worker.name}: {obj} (attempt {attempt}/{args.max_attempts})", flush=True)

            if not active:
                continue
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                worker, obj, attempt = active.pop(future)
                available.append(worker)
                try:
                    returncode, log_path = future.result()
                except Exception as exc:
                    returncode, log_path = 999, f"scheduler exception: {exc!r}"
                event = jobs[obj]["history"][-1]
                event.update({"finished_utc": _utc_now(), "returncode": returncode, "log": log_path})
                if returncode == 0 and _completed(shared_root / str(jobs[obj]["output_root_rel"]), obj):
                    jobs[obj]["status"] = "completed"
                    print(f"[done] {worker.name}: {obj}", flush=True)
                elif attempt < args.max_attempts:
                    jobs[obj]["status"] = "pending"
                    pending.append(obj)
                    print(f"[retry] {worker.name}: {obj} (returncode={returncode})", flush=True)
                else:
                    jobs[obj]["status"] = "failed"
                    print(f"[failed] {worker.name}: {obj}; see {log_path}", flush=True)
                with state_lock:
                    state["updated_utc"] = _utc_now()
                    _atomic_json(state_path, state)

    completed = [obj for obj in objects if jobs[obj]["status"] == "completed"]
    failed = [obj for obj in objects if jobs[obj]["status"] == "failed"]
    print(f"[summary] completed={len(completed)} failed={len(failed)} state={state_path}")
    if failed:
        print("[summary] failed objects:", " ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
