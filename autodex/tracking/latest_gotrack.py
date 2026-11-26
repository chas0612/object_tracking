"""Streaming capture bridge to the current MV-GoTrack tracker API.

Run in a separate process from the legacy runtime: both private repositories use
top-level Python packages named model, utils, scripts and evaluation.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from autodex.tracking.gotrack_runtime import checkpoint_path


@contextmanager
def working_directory(path):
    import os
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def homogeneous(value):
    if isinstance(value, dict):
        value = value.get("extrinsics_undistort", value.get("extrinsics"))
    out = np.asarray(value, dtype=np.float64)
    if out.shape == (3, 4):
        out = np.vstack((out, [0, 0, 0, 1]))
    if out.shape != (4, 4) or not np.isfinite(out).all():
        raise ValueError("Expected a finite 3x4 or 4x4 transform")
    return out


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def frame_indices(total, begin, end, order, max_frames):
    last = total - 1 if end < 0 else min(end, total - 1)
    # Preserve the legacy max-frames convention: exclusive original frame bound.
    if max_frames > 0:
        last = min(last, max_frames - 1)
    if begin < 0 or begin > last:
        raise ValueError(f"Empty or invalid tracking interval {begin}..{last}")
    return range(begin, last + 1) if order == "forward" else range(last, begin - 1, -1)


class CaptureReader:
    """Bounded video decoding with inline remapping and exact resize/K agreement."""
    def __init__(self, root, camera_ids, resize_scale):
        self.root, self.camera_ids = Path(root), camera_ids
        if not 0 < resize_scale <= 1:
            raise ValueError("input-resize-scale must be in (0, 1]")
        intr = json.loads((self.root / "cam_param/intrinsics.json").read_text())
        extr = json.loads((self.root / "cam_param/extrinsics.json").read_text())
        inline = (self.root / "inline_undistort.json").is_file()
        self.captures, self.maps, self.sizes, ks, es = [], [], [], [], []
        self.total = None
        self.fps = None
        try:
            for cid in camera_ids:
                paths = [self.root / sub / f"{cid}{suffix}"
                         for sub in (("videos",) if inline else ("undistorted_video", "videos"))
                         for suffix in (".avi", ".mp4", ".mov", ".mkv")]
                video = next((p for p in paths if p.is_file()), None)
                if video is None:
                    raise FileNotFoundError(f"No input video for {cid}")
                cap = cv2.VideoCapture(str(video))
                self.captures.append(cap)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open {video}")
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.total = n if self.total is None else min(self.total, n)
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                self.fps = fps if self.fps is None else self.fps
                w, h = int(cap.get(3)), int(cap.get(4))
                record = intr[cid]
                # Capture calibration names this K explicitly; do not prefer a
                # generic intrinsics field that could describe the distorted image.
                K = np.asarray(record["intrinsics_undistort"], dtype=np.float64)
                maps = None
                if inline or video.parent.name == "videos":
                    maps = cv2.initUndistortRectifyMap(
                        np.asarray(record["original_intrinsics"], dtype=np.float64),
                        np.asarray(record.get("dist_params", []), dtype=np.float64),
                        None, K, (w, h), cv2.CV_32FC1)
                    maps = cv2.convertMaps(*maps, cv2.CV_16SC2)
                rw, rh = max(1, round(w * resize_scale)), max(1, round(h * resize_scale))
                K = K.copy()
                K[0] *= rw / w
                K[1] *= rh / h
                self.maps.append(maps)
                self.sizes.append((rw, rh))
                ks.append(K)
                es.append(homogeneous(extr[cid]))
            self.K, self.T_c_w = np.stack(ks), np.stack(es)
        except BaseException:
            self.close()
            raise
        self._cache, self._next = {}, None

    def _decode(self):
        images = []
        for cap, maps, size in zip(self.captures, self.maps, self.sizes):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to decode a synchronized capture frame")
            if maps is not None:
                frame = cv2.remap(frame, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return images

    def read(self, index, reverse=False):
        if reverse:
            if index not in self._cache:
                start = max(0, index - 7)
                self._cache.clear()
                for cap in self.captures:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for frame in range(start, index + 1):
                    self._cache[frame] = self._decode()
                self._next = None
            return self._cache.pop(index)
        if self._next != index:
            for cap in self.captures:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        images = self._decode()
        self._next = index + 1
        return images

    def close(self):
        for cap in self.captures:
            cap.release()


def adapter_config(root, args, views):
    name = "irls" if args.backend == "irls" else "trifit"
    cfg = json.loads((root / f"configs/tracker/{name}.json").read_text())
    cfg.update(official_config_path=str(root / "configs/tracker/runner_ycbv.json"),
               checkpoint=str(Path(args.checkpoint_path).resolve()), gpu="0",
               output_dir=str(Path(args.output_root).resolve() / "adapter_runtime"),
               diameter_source="compute", anchor_bank_size=args.num_anchors,
               observer_compile=args.observer_compile)
    # IRLS tables must fit the rig, including rigs with more than four cameras.
    cfg.update(infoform_gpu_v_pad=views, infoform_gpu_a_pad=max(256, args.num_anchors))
    official_override = getattr(args, "official_config", None)
    if official_override:
        path = Path(official_override).expanduser().resolve()
        # Copy the resolved experiment input into its output. Never edit the
        # pinned runtime's shared config; later comparisons must be reproducible.
        raw = json.loads(path.read_text())
        snapshot = Path(args.output_root).resolve() / "official_config_resolved.json"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        write_json(snapshot, raw)
        cfg.update(official_config_path=str(snapshot),
                   official_config_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    return cfg


def joint_observation_overrides(args):
    """Per-run articulated overrides; absence preserves upstream defaults."""
    cfg = {}
    if getattr(args, "joint_residual_ratio", None) is not None:
        cfg["theta_residual_ratio"] = args.joint_residual_ratio
    if getattr(args, "min_moving_anchors", None) is not None:
        cfg["min_moving_anchors"] = args.min_moving_anchors
    return cfg


def world_record(index, fps, pose, result, latency):
    pose = homogeneous(pose)
    health = result.health or {}
    record = dict(frame_index=int(index), time_sec=index / fps if fps else None,
                status=result.status, stage="multiview_" + result.fusion_backend,
                pose_world=pose.tolist(), rotation_world=pose[:3, :3].tolist(),
                translation_world_m=pose[:3, 3].tolist(),
                translation_world_mm=(pose[:3, 3] * 1000).tolist(),
                fit_status=result.fit_status, fit_score=result.fit_score, score=result.score,
                latency_ms=latency, components_ms=result.components_ms, health=health,
                num_inlier_anchors=health.get("num_inlier_anchors"),
                mean_anchor_fit_residual_mm=health.get("mean_residual_mm"))
    for key in ("joint_value", "joint_type", "theta_rad", "theta_deg", "theta_observed",
                "theta_rejected", "theta_clamped", "joint_source"):
        if key in health:
            record[key] = health[key]
    return record


def run_capture(args):
    root = Path(args.gotrack_dir).resolve()
    from autodex.tracking.gotrack_runtime import LATEST_SHA
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if sha != LATEST_SHA:
        raise RuntimeError(f"This bridge is validated against {LATEST_SHA}; selected runtime is {sha}")
    # Reject mixing private module versions rather than depending on import order.
    for name in ("model", "utils", "evaluation", "scripts"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f"{name} was already imported from another runtime; use a subprocess")
    sys.path.insert(0, str(root))
    from evaluation.dexycb_tracking.adapters import GoTrackTrackingAdapter
    from utils.preregistration import create, digest
    import torch

    output = Path(args.output_root).resolve() / args.object_name
    output.mkdir(parents=True, exist_ok=False)
    reader = CaptureReader(args.input_root, args.camera_ids, args.input_resize_scale)
    records = []
    started = time.perf_counter()
    try:
        indices = list(frame_indices(reader.total, args.frame_begin, args.frame_end,
                                     args.frame_order, args.max_frames))
        init_root = Path(args.init_pose_source)
        init_path = init_root / "frame_poses" / f"{args.camera_ids[0]}.json"
        if not init_path.is_file():
            init_path = init_root / f"{args.camera_ids[0]}.json"
        payload = json.loads(init_path.read_text())
        candidates = payload.get("frames", payload) if isinstance(payload, dict) else payload
        seed = next((r for r in candidates if int(r["frame_index"]) == indices[0]), None)
        if seed is None:
            raise ValueError(f"No initialization for start frame {indices[0]}")
        prior = homogeneous(seed["pose_world"])
        cfg = adapter_config(root, args, len(args.camera_ids))
        articulated = getattr(args, "articulation_json", None)
        adapter_cls = GoTrackTrackingAdapter
        if articulated:
            from evaluation.dexycb_tracking.articulated import ArticulatedGoTrackTrackingAdapter, load_joint_spec
            adapter_cls = ArticulatedGoTrackTrackingAdapter
            spec = load_joint_spec(articulated)
            hinge = spec["joint_type"] == "revolute"
            degree_cap = args.theta_extrapolate_max_deg
            metre_cap = args.joint_extrapolate_max
            if (hinge and metre_cap is not None) or (not hinge and degree_cap is not None):
                raise ValueError("Joint extrapolation units do not match the joint type")
            anchors = json.loads(Path(args.joint_anchor_json).read_text()) if args.joint_anchor_json else {}
            if anchors and any(isinstance(v, dict) for v in anchors.values()):
                if set(anchors) != {args.object_name}:
                    raise ValueError("Joint anchor object names do not match this run")
                anchors = anchors[args.object_name]
            initial_joint = args.init_joint_value
            if initial_joint is None:
                initial_joint = seed.get("joint_value", seed.get("theta_rad", 0))
            cfg.update(articulation_json=str(Path(articulated).resolve()),
                       articulated_num_anchors=args.num_anchors, init_joint_value=float(initial_joint),
                       articulated_anchor_bank_path=args.anchor_bank_path, joint_anchors=anchors,
                       joint_extrapolate_max=float(np.radians(degree_cap or 0) if hinge else (metre_cap or 0)))
            # Optional per-run observation gates, not changes to the runtime or
            # its defaults. Rigid trackers never consume these parameters.
            cfg.update(joint_observation_overrides(args))
            if getattr(args, 'body_flow_bootstrap', False): cfg['body_flow_bootstrap'] = True
            stability = getattr(args, "articulated_stability_profile", None)
            if stability is not None:
                cfg['articulated_stability'] = json.loads(Path(stability).read_text())
                write_json(output / 'articulated_stability_profile.json', cfg['articulated_stability'])
            step_cap = getattr(args, "max_joint_step_deg", None)
            if step_cap is not None:
                if not hinge:
                    raise ValueError("--max-joint-step-deg requires a revolute joint")
                cfg["max_joint_step_deg"] = float(step_cap)
        registration = create(output / "registration.json", "capture runtime migration",
                              dict(config=cfg, camera_ids=args.camera_ids, frame_indices=indices,
                                   mesh=str(Path(args.mesh_path).resolve()), init_pose=prior.tolist(),
                                   protocol="closed_loop", reference_is_ground_truth=False,
                                   runtime_sources={str(p.relative_to(root)):
                                       hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                                           root / "evaluation/dexycb_tracking/adapters.py",
                                           root / "evaluation/dexycb_tracking/articulated.py",
                                           root / "utils/renderer_nvdiffrast.py",
                                           root / "utils/anchor_bank.py", root / "utils/anchor_tracking.py",
                                           root / "utils/multiview_geometry.py") if p.is_file()},
                                   bridge_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   joint_step_guard_sha256=(hashlib.sha256(
                                       Path(__file__).with_name("joint_step_guard.py").read_bytes()).hexdigest()
                                       if cfg.get("max_joint_step_deg") is not None else None),
                                   articulated_stability_sha256=(hashlib.sha256(
                                       Path(__file__).with_name('articulated_stability.py').read_bytes()).hexdigest()
                                       if 'articulated_stability' in cfg else None),
                                   robust_rays_sha256=(hashlib.sha256(
                                       Path(__file__).with_name('robust_rays.py').read_bytes()).hexdigest()
                                       if cfg.get('articulated_stability',{}).get('robust_ray_consensus') else None),
                                   body_flow_bootstrap_sha256=(hashlib.sha256(
                                       Path(__file__).with_name('body_flow_bootstrap.py').read_bytes()).hexdigest()
                                       if cfg.get('body_flow_bootstrap') else None)))
        write_json(output / "adapter_config.json", cfg)
        with working_directory(root):
            adapter = adapter_cls(cfg)
            t = time.perf_counter()
            adapter.initialize(1, Path(args.mesh_path), reader.K, reader.T_c_w)
            stability_backend = None
            if articulated and 'articulated_stability' in cfg:
                from autodex.tracking.articulated_stability import StableArticulatedBackend
                stability_backend = StableArticulatedBackend(
                    adapter, cfg['articulated_stability'], output / 'stability_diagnostics')
                adapter._fusion_backend = stability_backend
            if articulated and getattr(args, "max_joint_step_deg", None) is not None:
                from autodex.tracking.joint_step_guard import JointStepGuard
                adapter._fusion_backend = JointStepGuard(
                    adapter._fusion_backend, adapter, np.radians(args.max_joint_step_deg))
            initialize_sec = time.perf_counter() - t
            bank = adapter._anchor_cache[1]
            # The upstream producer bank has no mesh-face/barycentric fields.
            # Preserve its real fields without inventing legacy sampler data.
            np.savez_compressed(output / "anchor_bank.npz", **{
                k: (np.asarray(json.dumps(json_value(v), sort_keys=True)) if k == "metadata"
                    else np.asarray(v)) for k, v in bank.items()})
            seed_images = reader.read(indices[0], args.frame_order == "reverse")
            if articulated:
                adapter.set_frame(indices[0])
            t = time.perf_counter()
            for _ in range(args.warmup_steps):
                adapter.step(seed_images, reader.K, reader.T_c_w, prior)
            adapter.synchronize()
            warmup_sec = time.perf_counter() - t
            adapter.reset()
            if stability_backend is not None:
                stability_backend.reset_support_history()
            bootstrap = None
            if cfg.get('body_flow_bootstrap', False):
                from autodex.tracking.body_flow_bootstrap import BodyFlowBootstrap
                bootstrap = BodyFlowBootstrap(adapter, reader.K, reader.T_c_w,
                    visible_only=cfg.get('articulated_stability',{}).get('body_flow_visible_only',False),
                    min_views=int(cfg.get('articulated_stability',{}).get('min_view_support',2)))
            for step, index in enumerate(indices):
                if articulated:
                    adapter.set_frame(index)
                images = seed_images if step == 0 else reader.read(index, args.frame_order == "reverse")
                flow_health = {}
                if bootstrap is not None:
                    proposal, flow_health = bootstrap.propose(images)
                    if proposal is not None: prior = proposal
                if articulated and 'articulated_stability' in cfg:
                    adapter.flow_prior_health=flow_health.copy()
                adapter.synchronize()
                t = time.perf_counter()
                for _ in range(max(1, args.first_frame_num_iters) if step == 0 else 1):
                    result = adapter.step(images, reader.K, reader.T_c_w, prior)
                    prior = homogeneous(result.pose_T_w_o)
                if stability_backend is not None:
                    stability_backend.commit_support(result.status,result.health)
                result.health.update(flow_health)
                if bootstrap is not None:
                    bootstrap.update(images, prior)
                adapter.synchronize()
                latency = (time.perf_counter() - t) * 1000
                records.append(world_record(index, reader.fps, prior, result, latency))
                if step % 25 == 0 or step == len(indices) - 1:
                    write_json(output / "world_pose_records.json", records)
                    print(f"[latest-gotrack] frame={index} status={result.status} "
                          f"fit={result.fit_status} step_ms={latency:.1f}", flush=True)
        times = [r["latency_ms"] for r in records[1:]]
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        write_json(output / "runtime_manifest.json", dict(
            upstream_sha=sha, registration_digest=digest(registration), camera_ids=args.camera_ids,
            initialize_sec=initialize_sec, warmup_sec=warmup_sec,
            wall_sec=time.perf_counter() - started, frames=len(records),
            mean_step_ms=float(np.mean(times)) if times else None,
            p95_step_ms=float(np.percentile(times, 95)) if times else None,
            fallback_frames=sum(r["status"] != "ok" for r in records),
            timing_note="step includes GPU synchronization; excludes decode/remap/resize, warmup and seed bootstrap",
            deploy_manifest=adapter.deploy_manifest))
        # Existing articulated merge/status tools use this completion marker.
        write_json(output.parent / "multi_object_stage_c_summary.json", dict(
            runtime="latest", objects=[args.object_name], frames=len(records),
            backend=args.backend, upstream_sha=sha))
    finally:
        if records:
            write_json(output / "world_pose_records.json", records)
        reader.close()
