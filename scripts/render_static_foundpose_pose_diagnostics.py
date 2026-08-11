#!/usr/bin/env python3
"""Render static FoundPose poses as source/mesh-only/overlay diagnostics.

The renderer uses the calibrated undistorted image already saved by the static
pipeline.  Mesh-only panels preserve OBJ texture when present and otherwise use
shaded vertex colour, making rotations easier to inspect than a flat silhouette.
Schedules are ordered oldest-to-newest and later completed tasks take priority.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from autodex.perception.silhouette import SilhouetteOptimizer  # noqa: E402
from src.visualization.overlay_object_video_single import load_cam_param  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_completed(schedules: list[Path]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for schedule in schedules:
        for path in sorted((schedule / "tasks").glob("*.json")):
            task = _read_json(path)
            if (
                task.get("status") == "completed"
                and task.get("attempt_dir")
                and task.get("episode_rel")
            ):
                latest[str(task["episode_rel"])] = task
    return latest


def _safe_component(value: str) -> str:
    rendered = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return rendered.strip("._") or "unknown"


def _output_relative(task: dict[str, Any], suffix: str) -> Path:
    parts = Path(str(task["episode_rel"])).parts
    operator = parts[-3] if len(parts) >= 3 else "unknown"
    return Path(_safe_component(operator)) / f"{_safe_component(str(task['task_id']))}_{suffix}.jpg"


def _resolve_task(latest: dict[str, dict[str, Any]], selector: str) -> dict[str, Any]:
    if selector in latest:
        return latest[selector]
    matches = [task for task in latest.values() if str(task.get("task_id")) == selector]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        episodes = sorted(str(task["episode_rel"]) for task in matches)
        raise KeyError(f"Ambiguous task {selector!r}; use one of these episode paths: {episodes}")
    raise KeyError(f"No completed task matching {selector!r}")


def _frame_dir(shared: Path, task: dict[str, Any]) -> Path:
    attempt = shared / task["attempt_dir"]
    candidates = sorted(attempt.glob("foundpose_frame_*"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one foundpose_frame_* under {attempt}")
    return candidates[0]


def _camera_with_suffix(frame_dir: Path, suffix: str) -> str:
    matches = sorted(path.stem for path in (frame_dir / "images").glob(f"*{suffix}.png"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one camera ending in {suffix!r} under {frame_dir}; got {matches}")
    return matches[0]


def _as_pose(value: object) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0.0, 0.0, 0.0, 1.0]])
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid pose shape/value: {pose.shape}")
    return pose


class TexturedRenderer:
    def __init__(self, mesh_path: Path):
        self.optimizer = SilhouetteOptimizer(str(mesh_path), device="cuda")
        from Utils import nvdiffrast_render
        self._render = nvdiffrast_render

    def render(self, pose_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray,
               height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        pose_camera = extrinsic @ pose_world
        pose_tensor = torch.as_tensor(
            pose_camera, device="cuda", dtype=torch.float32,
        ).reshape(1, 4, 4)
        rgb, _, _ = self._render(
            K=np.asarray(intrinsic, dtype=np.float32), H=height, W=width,
            ob_in_cams=pose_tensor, glctx=self.optimizer.glctx,
            mesh_tensors=self.optimizer.mesh_tensors, use_light=True,
            w_ambient=0.75, w_diffuse=0.45,
        )
        render_rgb = (rgb[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
        mask = render_rgb.sum(axis=2) > 0
        return render_bgr, mask


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(output, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _panels(source: np.ndarray, render: np.ndarray, mask: np.ndarray,
            label: str, panel_width: int) -> np.ndarray:
    background = np.full_like(source, 210)
    background[mask] = render[mask]
    overlay = source.copy()
    overlay[mask] = (0.35 * source[mask] + 0.65 * render[mask]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 3, cv2.LINE_AA)
    height = round(source.shape[0] * panel_width / source.shape[1])
    cells = []
    for title, image in (("SOURCE", source), ("MESH ONLY", background), ("OVERLAY", overlay)):
        resized = cv2.resize(image, (panel_width, height), interpolation=cv2.INTER_AREA)
        cells.append(_label(resized, f"{label}  {title}"))
    return np.concatenate(cells, axis=1)


def _task_diagnostic(
    *, shared: Path, task: dict[str, Any], camera_suffix: str,
    output: Path, renderer: TexturedRenderer, panel_width: int,
) -> None:
    frame_dir = _frame_dir(shared, task)
    serial = _camera_with_suffix(frame_dir, camera_suffix)
    source = cv2.imread(str(frame_dir / "images" / f"{serial}.png"), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(frame_dir / "images" / f"{serial}.png")
    intrinsics, extrinsics = load_cam_param(shared / task["episode_rel"] / "cam_param")
    pose = _as_pose(np.load(shared / task["attempt_dir"] / "foundpose_init/init_pose_world.npy"))
    render, mask = renderer.render(
        pose, intrinsics[serial], extrinsics[serial], source.shape[0], source.shape[1],
    )
    sheet = _panels(source, render, mask, f"{task['task_id']}  cam={serial}", panel_width)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write {output}")


def _candidate_diagnostic(
    *, shared: Path, task: dict[str, Any], camera_suffix: str, bank_name: str,
    output: Path, renderer: TexturedRenderer, panel_width: int,
) -> None:
    frame_dir = _frame_dir(shared, task)
    serial = _camera_with_suffix(frame_dir, camera_suffix)
    source = cv2.imread(str(frame_dir / "images" / f"{serial}.png"), cv2.IMREAD_COLOR)
    intrinsics, extrinsics = load_cam_param(shared / task["episode_rel"] / "cam_param")
    init_dir = shared / task["attempt_dir"] / "foundpose_init"
    bank = _read_json(init_dir / bank_name)
    candidates = bank.get("candidates", [])
    if source is None or not candidates:
        raise RuntimeError("Candidate source image or candidate bank is empty")
    rows: list[np.ndarray] = []
    for rank, candidate in enumerate(candidates):
        pose = _as_pose(candidate["pose_world"])
        render, mask = renderer.render(
            pose, intrinsics[serial], extrinsics[serial], source.shape[0], source.shape[1],
        )
        score = float(candidate.get("mean_iou", float("nan")))
        source_name = str(candidate.get("source_serial", "unknown"))
        rows.append(_panels(
            source, render, mask,
            f"rank={rank} iou={score:.3f} {source_name} cam={serial}", panel_width,
        ))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-id", nargs="+", required=True,
                        help="Oldest-to-newest schedules.")
    parser.add_argument(
        "--runs-root-rel",
        default="object_tracking/campaigns/corl_rebuttal/foundpose_static_runs",
    )
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument("--objects", nargs="*", default=[])
    parser.add_argument("--candidate-task", default=None)
    parser.add_argument("--candidate-bank", default="global_coarse_bank.json")
    parser.add_argument("--camera-suffix", default="3282")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.panel_width < 96 or (not args.objects and not args.candidate_task):
        raise ValueError("Set --objects and/or --candidate-task; panel width must be >=96")
    shared = Path.home() / args.shared_root_rel
    runs_root = shared / args.runs_root_rel
    schedules = [runs_root / value for value in args.schedule_id]
    latest = _latest_completed(schedules)
    selected = [
        task for task in latest.values()
        if task.get("source_object", task.get("object_name")) in set(args.objects)
        or task.get("object_name") in set(args.objects)
    ]
    selected.sort(key=lambda task: str(task["task_id"]))
    candidate_task = _resolve_task(latest, args.candidate_task) if args.candidate_task else None
    print(f"tasks={len(selected)} candidate_task={args.candidate_task}")
    for task in selected:
        print(f"  {task['task_id']} mesh={task['mesh_rel']}")
    if args.dry_run:
        return 0

    output_dir = Path(args.output_dir).expanduser().resolve()
    renderers: dict[str, TexturedRenderer] = {}

    def renderer_for(task: dict[str, Any]) -> TexturedRenderer:
        mesh_rel = str(task["mesh_rel"])
        if mesh_rel not in renderers:
            renderers[mesh_rel] = TexturedRenderer(shared / mesh_rel)
        return renderers[mesh_rel]

    for task in selected:
        output = output_dir / _output_relative(task, f"cam_{args.camera_suffix}")
        if output.exists() and not args.overwrite:
            print(f"[skip] {output}")
            continue
        _task_diagnostic(
            shared=shared, task=task, camera_suffix=args.camera_suffix,
            output=output, renderer=renderer_for(task), panel_width=args.panel_width,
        )
        print(f"[wrote] {output}")

    if candidate_task is not None:
        task = candidate_task
        output = output_dir / _output_relative(
            task, f"{Path(args.candidate_bank).stem}_cam_{args.camera_suffix}",
        )
        if output.exists() and not args.overwrite:
            print(f"[skip] {output}")
        else:
            _candidate_diagnostic(
                shared=shared, task=task, camera_suffix=args.camera_suffix,
                bank_name=args.candidate_bank, output=output,
                renderer=renderer_for(task), panel_width=args.panel_width,
            )
            print(f"[wrote] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
