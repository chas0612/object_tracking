#!/usr/bin/env python3
"""Add coarse appearance-matched vertex colors to an ARCTIC scissors OBJ.

The simplified articulated mesh has no UV correspondence with ARCTIC's textured
mesh.  OBJ vertex colors keep the vertex and face order unchanged, which is
required by ``face_part_ids.npy`` and the articulated GoTrack anchor bank.
"""

from __future__ import annotations

import argparse
from pathlib import Path


# Linear RGB-like values consumed directly by trimesh/nvdiffrast.  The physical
# scissors in ep08 are nearly black at the handles and dark charcoal on the blades.
# Do not isolate the pivot: its small highlight is view-dependent, whereas a bright
# baked patch becomes an incorrect, persistent feature in every rendered view.
HANDLE_RGB = (0.055, 0.060, 0.065)
BLADE_RGB = (0.135, 0.145, 0.155)


def color_for_vertex(x: float, handle_end_x: float):
    if x < handle_end_x:
        return "handle", HANDLE_RGB
    return "blade", BLADE_RGB


def colorize(source: Path, output: Path, handle_end_x: float) -> dict[str, int]:
    counts = {"handle": 0, "blade": 0}
    lines: list[str] = []
    for raw in source.read_text().splitlines():
        fields = raw.split()
        if len(fields) >= 4 and fields[0] == "v":
            x, y, z = map(float, fields[1:4])
            label, rgb = color_for_vertex(x, handle_end_x)
            counts[label] += 1
            lines.append(
                f"v {x:.8f} {y:.8f} {z:.8f} "
                f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}"
            )
        else:
            lines.append(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--handle-end-x",
        type=float,
        default=-0.020,
        help="Vertices below this canonical x are colored as handle (metres).",
    )
    args = parser.parse_args()
    counts = colorize(args.input, args.output, args.handle_end_x)
    print(f"wrote {args.output}")
    print("vertex colors: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
