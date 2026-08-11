#!/usr/bin/env python3
"""Finalize reviewed CORL grasp poses under the consumer-facing filename."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_stream, temporary.open("wb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_pose(path: Path) -> None:
    with np.load(path) as archive:
        if archive.files != ["frame_0"]:
            raise ValueError(f"Expected only frame_0 in {path}: {archive.files}")
        pose = np.asarray(archive["frame_0"], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid pose matrix in {path}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"Invalid homogeneous row in {path}")
    if not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=2e-3):
        raise ValueError(f"Invalid rotation determinant in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument("--source-manifest-rel", nargs="+", required=True)
    parser.add_argument("--output-name", default="object_6d_pose.npz")
    parser.add_argument("--expected-tasks", type=int, default=60)
    parser.add_argument(
        "--manifest-rel",
        default="capture/corl_rebuttal/object_6d_pose_final_manifest.json",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    shared = Path.home() / args.shared_root_rel
    final_records: list[dict[str, object]] = []
    seen: set[Path] = set()
    for manifest_rel in args.source_manifest_rel:
        manifest_path = shared / manifest_rel
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["records"]:
            source = shared / str(record["target_rel"])
            target = source.parent / args.output_name
            if target in seen:
                raise ValueError(f"Duplicate final target: {target}")
            seen.add(target)
            if not source.is_file():
                raise FileNotFoundError(source)
            _validate_pose(source)
            final_records.append({
                "episode_rel": record["episode_rel"],
                "object": record["object"],
                "source_manifest_rel": manifest_rel,
                "source_rel": str(source.relative_to(shared)),
                "source_sha256": _sha256(source),
                "target_rel": str(target.relative_to(shared)),
            })
    if len(final_records) != args.expected_tasks:
        raise RuntimeError(f"Expected {args.expected_tasks} tasks, found {len(final_records)}")
    existing = [shared / str(record["target_rel"]) for record in final_records
                if (shared / str(record["target_rel"])).exists()]
    if existing:
        raise FileExistsError(f"Final targets already exist; first: {existing[0]}")
    print(f"tasks={len(final_records)}")
    if not args.write:
        for record in final_records:
            print(f"[audit] {record['target_rel']} <- {record['source_rel']}")
        return 0

    for record in final_records:
        source = shared / str(record["source_rel"])
        target = shared / str(record["target_rel"])
        _atomic_copy(source, target)
        _validate_pose(target)
        record["target_sha256"] = _sha256(target)
        if record["target_sha256"] != record["source_sha256"]:
            raise RuntimeError(f"Hash mismatch after copy: {target}")
        print(f"[wrote] {target}")
    output_manifest = shared / args.manifest_rel
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_rels": args.source_manifest_rel,
        "output_name": args.output_name,
        "tasks": len(final_records),
        "records": final_records,
    }
    _atomic_json(output_manifest, payload)
    print(f"manifest={output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
