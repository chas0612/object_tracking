#!/usr/bin/env python3
"""Initialize a world pose from calibrated stereo and refine it against SAM3 masks."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import sysconfig
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RIG_PAIRS = (
    ("23180202", "22645021"),
    ("23263780", "22684755"),
    ("22645026", "22645029"),
    ("25452066", "26256735"),
)
BACKEND = "stereo_foundationpose"
DEFAULT_RUNTIME_ROOT = "~/object-6d-tracking"
ENGINE_REL = "thirdparty/FoundationStereo/pretrained_models/foundation_stereo.plan"
THIRD_PARTY = REPO_ROOT / "autodex/perception/thirdparty"
FOUNDATIONPOSE_RUNS = ("2024-01-11-20-02-45", "2023-10-28-18-33-37")


def validate_pairs(pairs):
    pairs = tuple(tuple(pair) for pair in pairs)
    if not 1 <= len(pairs) <= 4 or len(set(pairs)) != len(pairs):
        raise ValueError("Specify between one and four distinct stereo pairs.")
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise ValueError("Each stereo pair needs different left/right camera IDs.")
    return pairs


def load_calibration(capture_dir):
    root = capture_dir / "cam_param"
    intrinsic = json.loads((root / "intrinsics.json").read_text())
    extrinsic = json.loads((root / "extrinsics.json").read_text())
    if not isinstance(intrinsic, dict) or not isinstance(extrinsic, dict):
        raise ValueError("Camera calibration files must map camera IDs to entries.")
    return intrinsic, extrinsic


def add_scheduler_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--init-backend", choices=("foundpose", BACKEND), default="foundpose")
    parser.add_argument("--stereo-pair", nargs=2, action="append", metavar=("LEFT", "RIGHT"))
    parser.add_argument("--stereo-env", default="object_6d")
    parser.add_argument("--stereo-runtime-root", default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--stereo-engine", help="Optional worker-local FoundationStereo TensorRT plan path.")
    parser.add_argument("--stereo-silhouette-iters", type=int, default=60)


def scheduler_config(args: argparse.Namespace) -> dict:
    if args.stereo_silhouette_iters < 1:
        raise ValueError("--stereo-silhouette-iters must be positive")
    return {
        "pairs": validate_pairs(args.stereo_pair or RIG_PAIRS), "env": args.stereo_env,
        "runtime_root": args.stereo_runtime_root, "engine": args.stereo_engine,
        "silhouette_iters": args.stereo_silhouette_iters,
    }


def stereo_init_command(
    config: dict, capture: Path, frame_dir: Path, mesh: Path,
    object_name: str, output: Path, frame_index: int, *, conda: str | None = None,
) -> list[str]:
    command = [conda or str(Path.home() / "anaconda3/bin/conda"), "run", "--no-capture-output",
               "-n", config["env"], "python", "-u", "src/process/stereo_foundationpose_init_capture.py",
               "--capture-dir", str(capture), "--frame-dir", str(frame_dir), "--mesh", str(mesh),
               "--object-name", object_name, "--output-dir", str(output), "--frame-index", str(frame_index),
               "--stereo-runtime-root", config["runtime_root"], "--silhouette-iters", str(config["silhouette_iters"])]
    if config["engine"]:
        command.extend(["--stereo-engine", config["engine"]])
    for left, right in config["pairs"]:
        command.extend(["--stereo-pair", left, right])
    return command


def foundationpose_assets(runtime_root: Path) -> tuple[Path, Path]:
    vendored = THIRD_PARTY / "FoundationPose"
    standalone = runtime_root / "thirdparty/FoundationPose"
    weights_locations = (vendored / "weights", THIRD_PARTY / "weights/foundationpose", standalone / "weights")
    weights = next((root for root in weights_locations if all(
        (root / run / name).is_file()
        for run in FOUNDATIONPOSE_RUNS for name in ("model_best.pth", "config.yml")
    )), None)
    if weights is None:
        raise FileNotFoundError(f"FoundationPose weight sets not found in {weights_locations}")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extension_locations = tuple(root / "mycpp/build" / f"mycpp{suffix}" for root in (vendored, standalone))
    extension = next((path for path in extension_locations if path.is_file()), None)
    if extension is None:
        raise FileNotFoundError(f"Built FoundationPose mycpp not found in {extension_locations}")
    return weights, extension


class StereoInitializer:
    """One TRT engine and one set of FoundationPose weights per init worker."""

    def __init__(self, runtime_root: Path, engine: Path | None):
        import nvdiffrast.torch as dr

        weights, extension = foundationpose_assets(runtime_root)
        sys.path[:0] = [str(extension.parent), str(THIRD_PARTY / "FoundationPose")]
        import estimater
        if estimater.mycpp is None:
            import mycpp
            estimater.mycpp = mycpp
        from estimater import PoseRefinePredictor, ScorePredictor

        source = runtime_root / "tracking/core/depth_estimator.py"
        engine = engine if engine is not None else runtime_root / ENGINE_REL
        for path in (source, engine):
            if not path.is_file():
                raise FileNotFoundError(f"FoundationStereo runtime file missing: {path}")
        spec = importlib.util.spec_from_file_location("capture_stereo_depth", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import FoundationStereo runtime: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.stereo = module.StereoDepthEstimator(
            engine_path=str(engine), K_rect=np.eye(3), baseline=1.0,
        )
        shapes = [tuple(entry["shape"]) for entry in self.stereo.engine.inputs]
        if (len(shapes) != 2 or len(shapes[0]) != 4 or shapes[0] != shapes[1]
                or shapes[0][:2] != (1, 3) or min(shapes[0]) <= 0):
            raise ValueError(f"Expected a static FoundationStereo RGB engine, got {shapes}")
        _, _, height, width = shapes[0]
        self.size = width, height
        self.stereo.width, self.stereo.height = self.size
        self.engine = engine
        self.scorer = ScorePredictor(weights_root=weights)
        self.refiner = PoseRefinePredictor(weights_root=weights)
        self.weights_root = weights
        self.extension = extension
        self.context = dr.RasterizeCudaContext()
        self.mesh_path = None

    def set_mesh(self, mesh_path):
        from estimater import FoundationPose
        from autodex.perception.silhouette import SilhouetteOptimizer

        if self.mesh_path == mesh_path:
            return
        # Release the preceding object's GPU resources before binding a new one.
        self.mesh_path = None
        self.pose = self.silhouette = None
        self.silhouette = SilhouetteOptimizer(str(mesh_path))
        mesh = self.silhouette.mesh
        self.pose = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=self.scorer, refiner=self.refiner, glctx=self.context,
            debug=0, debug_dir="/tmp/foundationpose_capture_debug",
        )
        # The vendored estimator exports a temporary OBJ during construction.
        # Its register path uses mesh_tensors directly.
        temporary_mesh = Path(self.pose.mesh_path)
        for suffix in (".obj", ".mtl", "_0.png"):
            temporary_mesh.with_name(temporary_mesh.stem + suffix).unlink(missing_ok=True)
        self.mesh_path = mesh_path

    def propose(self, left_rgb, right_rgb, mask, left_k, right_k, left_extrinsic, right_extrinsic):
        import cv2

        if left_rgb.shape != right_rgb.shape or left_rgb.shape[:2] != mask.shape:
            raise ValueError("Stereo images and left mask have incompatible dimensions.")
        height, width = mask.shape
        left_to_right = right_extrinsic @ np.linalg.inv(left_extrinsic)
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            left_k, None, right_k, None, (width, height),
            left_to_right[:3, :3], left_to_right[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=1, newImageSize=self.size,
        )
        if abs(p2[0, 3]) < 1e-6 or abs(p2[1, 3]) > abs(p2[0, 3]):
            raise ValueError("Stereo pair has no usable horizontal baseline.")
        left_map = cv2.initUndistortRectifyMap(left_k, None, r1, p1[:, :3], self.size, cv2.CV_32FC1)
        right_map = cv2.initUndistortRectifyMap(right_k, None, r2, p2[:, :3], self.size, cv2.CV_32FC1)
        left = cv2.remap(left_rgb, *left_map, cv2.INTER_LINEAR)
        right = cv2.remap(right_rgb, *right_map, cv2.INTER_LINEAR)
        flip = p2[0, 3] > 0
        inputs = (left[:, ::-1], right[:, ::-1]) if flip else (left, right)
        _, disparity = self.stereo.estimate(*inputs)
        if flip:
            disparity = -disparity[:, ::-1].copy()
        depth = cv2.reprojectImageTo3D(disparity.astype(np.float32), q)[..., 2].copy()
        y, x = np.indices(depth.shape, dtype=np.float32)
        right_visible = ((right_map[0] >= 0) & (right_map[0] < width - 1)
                         & (right_map[1] >= 0) & (right_map[1] < height - 1))
        overlap = cv2.remap(right_visible.astype(np.uint8), x - disparity.astype(np.float32), y, cv2.INTER_NEAREST) > 0
        left_visible = ((left_map[0] >= 0) & (left_map[0] < width - 1)
                        & (left_map[1] >= 0) & (left_map[1] < height - 1))
        valid = left_visible & overlap & np.isfinite(depth) & (depth > 0.05) & (depth <= 5.0)
        depth[~valid] = 0
        rect_mask = cv2.remap(mask.astype(np.uint8), *left_map, cv2.INTER_NEAREST) > 0
        valid_depth = depth[rect_mask & (depth > 0)]
        coverage = len(valid_depth) / max(1, int(rect_mask.sum()))
        if len(valid_depth) < 50 or coverage < 0.25:
            raise ValueError(f"Insufficient depth in left mask ({len(valid_depth)} pixels, {coverage:.1%}).")

        # FoundationPose's scorer warps each hypothesis to image resolution.
        # Crop the RGB-D and shift the principal point to bound that allocation.
        ys, xs = np.where(rect_mask)
        pad = max(24, int(max(np.ptp(xs), np.ptp(ys)) * 0.35))
        x0, x1 = max(0, int(xs.min()) - pad), min(self.size[0], int(xs.max()) + pad + 1)
        y0, y1 = max(0, int(ys.min()) - pad), min(self.size[1], int(ys.max()) + pad + 1)
        roi = np.s_[y0:y1, x0:x1]
        k = p1[:, :3].astype(np.float32)
        k[0, 2] -= x0
        k[1, 2] -= y0
        rect_from_left = np.eye(4)
        rect_from_left[:3, :3] = r1
        self.pose.pose_last = None
        rect_pose = self.pose.register(
            K=k, rgb=left[roi].copy(), depth=depth[roi].copy(),
            ob_mask=rect_mask[roi].copy(), iteration=5,
        )
        world = np.linalg.inv(rect_from_left @ left_extrinsic) @ rect_pose
        if self.pose.pose_last is None or not np.isfinite(world).all() or rect_pose[2, 3] <= 0:
            raise ValueError("FoundationPose returned an invalid camera pose.")
        return world, coverage


@lru_cache(maxsize=1)
def get_stereo_initializer(runtime_root, engine):
    return StereoInitializer(runtime_root, engine)


def initialize(capture_dir, frame_dir, mesh, object_name, output_dir, frame_index,
               pairs, runtime_root, engine, silhouette_iters):
    import cv2
    from src.process.foundpose_init_capture import _as_4x4
    from src.process.gotrack_capture import _video_paths_by_stem, _video_timing

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    metadata_path = frame_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("frame_index") != frame_index or not metadata.get("undistort_frame"):
            raise ValueError(f"Expected undistorted SAM3 inputs for video frame {frame_index}: {frame_dir}")
    intr_raw, extr_raw = load_calibration(capture_dir)
    camera_ids = set(intr_raw) & set(extr_raw)
    masks = {}
    for path in sorted((frame_dir / "masks").glob("*.png")):
        if path.stem not in camera_ids:
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and (mask > 127).any():
            masks[path.stem] = mask > 127
    if len(masks) < 2:
        raise RuntimeError("Stereo initialization requires at least two calibrated silhouette masks.")
    intrinsics = {cid: np.asarray(intr_raw[cid]["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
                  for cid in camera_ids}
    extrinsics = {cid: _as_4x4(extr_raw[cid]) for cid in camera_ids}
    videos = _video_paths_by_stem(capture_dir / "videos")
    candidates, pair_results = {}, []
    session = None
    for left, right in validate_pairs(pairs):
        label = f"{left}+{right}"
        report = {"left": left, "right": right}
        pair_results.append(report)
        if not {left, right} <= camera_ids & videos.keys():
            report["skipped"] = "missing_video_or_calibration"
            continue
        if left not in masks:
            report["skipped"] = "missing_left_mask"
            continue
        left_timing, right_timing = (_video_timing(videos[cid]) for cid in (left, right))
        if (left_timing["frames"] != right_timing["frames"]
                or frame_index >= left_timing["frames"]
                or min(left_timing["fps"], right_timing["fps"]) <= 0
                or not np.isclose(left_timing["fps"], right_timing["fps"], atol=0.01, rtol=0)):
            report["skipped"] = "stereo_video_timing_mismatch"
            report["video_timings"] = [left_timing, right_timing]
            continue
        rgb = [cv2.imread(str(frame_dir / "images" / f"{cid}.png")) for cid in (left, right)]
        if any(image is None for image in rgb):
            report["skipped"] = "missing_seed_rgb"
            continue
        if session is None:
            session = get_stereo_initializer(runtime_root, engine)
        session.set_mesh(mesh)
        started = time.perf_counter()
        try:
            pose, coverage = session.propose(
                *(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in rgb), masks[left],
                intrinsics[left], intrinsics[right], extrinsics[left], extrinsics[right],
            )
        except (ValueError, cv2.error) as exc:
            report["skipped"] = str(exc)
            continue
        candidates[label] = pose
        report.update(depth_coverage=coverage, seconds=time.perf_counter() - started)
        print(f"[stereo] {label} depth_coverage={coverage:.3f} sec={report['seconds']:.2f}", flush=True)
    if not candidates:
        raise RuntimeError(f"No stereo FoundationPose candidate: {pair_results}")

    from autodex.perception.pose_select import _resize_view_inputs, compute_cross_view_iou
    masks, intrinsics, height, width = _resize_view_inputs(masks, intrinsics, 512)
    if len(masks) < 2:
        raise RuntimeError("Need two calibrated masks with common resolution for silhouette refinement.")
    silhouette = session.silhouette
    views = [{"cam_id": cid, "mask": mask.astype(np.uint8) * 255,
              "K": intrinsics[cid].astype(np.float32), "extrinsic": extrinsics[cid]}
             for cid, mask in masks.items()]

    def score(pose):
        mean, per_view = compute_cross_view_iou(
            pose, masks, intrinsics, extrinsics, height, width,
            silhouette.glctx, silhouette.mesh_tensors,
        )
        values = np.sort(list(per_view.values()))
        trim = int(len(values) * 0.15)
        robust = values[trim:-trim].mean() if trim else mean
        return float(robust), float(mean)

    ranked = sorted(((score(pose), label, pose) for label, pose in candidates.items()),
                    key=lambda row: row[0], reverse=True)
    refine_count = 2 if len(ranked) > 1 and ranked[0][0][0] - ranked[1][0][0] <= 0.03 else 1
    results = []
    for idx, (before, label, seed) in enumerate(ranked):
        result = {"source_serial": label, "robust_iou_before": before[0], "mean_iou_before": before[1],
                  "pose_world_pre_silhouette": seed, "pose_world": seed, "robust_iou": before[0], "mean_iou": before[1]}
        if idx < refine_count:
            refined, loss = silhouette.optimize(seed, views, iters=silhouette_iters, lr=0.002, antialias=True)
            if np.isfinite(refined).all():
                after = score(refined)
                result.update(refined_robust_iou=after[0], refined_mean_iou=after[1], silhouette_loss=float(loss))
                if after >= before:
                    result.update(pose_world=refined, robust_iou=after[0], mean_iou=after[1])
        results.append(result)
    results.sort(key=lambda r: (r["robust_iou"], r["mean_iou"]), reverse=True)
    best = results[0]
    if best["robust_iou"] <= 0:
        raise RuntimeError("Stereo pose has no robust cross-view silhouette overlap.")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "init_pose_world.npy", best["pose_world"])
    np.save(output_dir / "init_pose_world_pre_silhouette.npy", best["pose_world_pre_silhouette"])
    np.savez(output_dir / "per_view_poses_world.npz", **candidates)
    bank = [{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in item.items()}
            for item in results]
    (output_dir / "candidate_bank.json").write_text(json.dumps({"selection_mode": BACKEND,
        "selected_rank": 0, "candidates": bank}, indent=2) + "\n")
    (output_dir / "result.json").write_text(json.dumps({
        "init_backend": BACKEND, "capture_dir": str(capture_dir), "frame_dir": str(frame_dir),
        "frame_index": frame_index, "mesh": str(mesh), "object_name": object_name,
        "stereo_engine": str(session.engine), "stereo_pairs": pair_results,
        "foundationpose_weights": str(session.weights_root), "foundationpose_mycpp": str(session.extension),
        "num_stereo_candidates": len(candidates), "num_input_views": len(masks),
        "iou_best_serial": best["source_serial"], "robust_iou": best["robust_iou"],
        "iou_mean": best["mean_iou"], "robust_iou_before": best["robust_iou_before"],
        "candidate_bank": "candidate_bank.json", "init_pose_world": best["pose_world"].tolist(),
    }, indent=2) + "\n")
    print(f"[done] {output_dir / 'init_pose_world.npy'} (robust IoU {best['robust_iou']:.3f})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-dir", "frame-dir", "mesh", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--stereo-pair", nargs=2, action="append", metavar=("LEFT", "RIGHT"))
    parser.add_argument("--stereo-runtime-root", type=Path, default=Path(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--stereo-engine", type=Path)
    parser.add_argument("--silhouette-iters", type=int, default=60)
    args = parser.parse_args()
    if args.frame_index < 0 or args.silhouette_iters < 1:
        parser.error("frame index must be nonnegative and silhouette iterations positive")
    initialize(args.capture_dir, args.frame_dir, args.mesh, args.object_name, args.output_dir,
               args.frame_index, args.stereo_pair or RIG_PAIRS, args.stereo_runtime_root.expanduser(),
               args.stereo_engine.expanduser() if args.stereo_engine else None, args.silhouette_iters)


if __name__ == "__main__":
    main()
