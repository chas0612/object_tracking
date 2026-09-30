#!/usr/bin/env python3
"""Create a pinned, patched private checkout without changing an existing checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
PINS = json.loads((REPO / "patches/MV-GoTrack-latest.json").read_text())


def git(*args):
    subprocess.run(["git", *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Authorized private clone or URL")
    parser.add_argument("--destination", default=str(REPO / "autodex/perception/thirdparty/MV-GoTrack-next"))
    parser.add_argument("--dependency-source", help="Local checkout containing external/{bop_toolkit,dinov2}")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    target = Path(args.destination).expanduser().absolute()
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    if target.exists() or target.is_symlink():
        parser.error(f"Destination already exists; choose a new directory: {target}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != PINS["checkpoint_sha256"]:
        parser.error("Checkpoint digest differs from the pinned original checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    git("clone", "--no-hardlinks", args.source, target)
    git("-C", target, "checkout", "--detach", PINS["upstream_sha"])
    if args.dependency_source:
        for name, sha in PINS["submodules"].items():
            dependency = target / "external" / name
            git("clone", "--no-hardlinks", Path(args.dependency_source).expanduser() / "external" / name, dependency)
            git("-C", dependency, "checkout", "--detach", sha)
    else:
        git("-C", target, "submodule", "update", "--init", "external/bop_toolkit", "external/dinov2")
    for name in PINS["patches"]:
        patch = REPO / "patches" / name
        git("-C", target, "apply", "--check", patch)
        git("-C", target, "apply", patch)
    (target / "checkpoints").mkdir(exist_ok=True)
    (target / "checkpoints/gotrack_checkpoint.pt").symlink_to(checkpoint)
    print(f"Prepared {target}; existing source and checkpoints were preserved.")


if __name__ == "__main__":
    main()
