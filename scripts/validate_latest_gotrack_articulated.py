#!/usr/bin/env python3
"""Validate a latest-runtime one-joint bridge on generated geometry (no capture GT)."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gotrack-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.gotrack_dir).resolve()
    sys.path[:0] = [str(repo), str(root)]
    import torch
    import trimesh
    from autodex.tracking.latest_gotrack import adapter_config, working_directory, write_json
    from evaluation.dexycb_tracking.articulated import ArticulatedGoTrackTrackingAdapter
    from multiview_solver.fusion_backends import FusionContext, FusionFrame
    from scripts.common.camera import build_camera_model
    from utils.anchor_tracking import pose_anchors_at_joint_angle
    from utils.renderer_base import RenderType

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    K = np.repeat(np.array([[[360, 0, 192], [0, 360, 192], [0, 0, 1]]], float), 4, 0)
    cameras = np.repeat(np.eye(4)[None], 4, 0)
    cameras[:, :2, 3] = [[-.13, 0], [.13, 0], [0, -.12], [0, .12]]
    pose = np.eye(4)
    pose[:3, 3] = [.015, -.012, .7]
    report = {}
    for kind, coordinate, limits in (("revolute", .35, [0, 1.2]), ("prismatic", .03, [0, .1])):
        directory = output / kind
        directory.mkdir()
        body = trimesh.creation.box([.14, .12, .08])
        moving = trimesh.creation.box([.13, .10, .015])
        moving.apply_translation([0, .045, .07])
        body.visual.face_colors = [220, 80, 60, 255]
        moving.visual.face_colors = [50, 180, 220, 255]
        body.export(directory / "body.ply")
        moving.export(directory / "moving.ply")
        trimesh.util.concatenate([body, moving]).export(directory / "fused.ply")
        spec = dict(parts=["body.ply", "moving.ply"], axis=[1, 0, 0], origin=[0, 0, .06],
                    joint_type=kind, **({"range_rad": limits} if kind == "revolute" else {"range": limits}))
        write_json(directory / "joint.json", spec)
        cfg = adapter_config(root, SimpleNamespace(backend="trifit", num_anchors=256,
            checkpoint_path=args.checkpoint, output_root=str(directory), observer_compile=True), 4)
        cfg.update(articulation_json=str(directory / "joint.json"), articulated_num_anchors=512,
                   init_joint_value=coordinate, wrong_surface_margin=0)
        with working_directory(root):
            tracker = ArticulatedGoTrackTrackingAdapter(cfg)
            tracker.initialize(1, directory / "fused.ply", K, cameras)
            bank = tracker._anchor_cache[1]
            source = bank["positions_o_external"]
            j = tracker.joint
            truth = pose_anchors_at_joint_angle(source, j["part_ids"], j["joint_axis_o"],
                        j["joint_origin_o"], coordinate, joint_type=kind)
            target = truth @ pose[:3, :3].T + pose[:3, 3]
            observations = {}
            for aid, point in enumerate(target):
                rows = []
                for cid, (intr, extr) in enumerate(zip(K, cameras)):
                    cam_point = extr[:3, :3] @ point + extr[:3, 3]
                    uv = intr @ cam_point
                    rows.append(dict(camera_id=str(cid), uv_curr=uv[:2] / uv[2],
                                     confidence=1., position_o=source[aid], valid_flag=True))
                observations[aid] = rows
            frame = FusionFrame(observations, {str(i): v for i, v in enumerate(K)},
                                {str(i): v for i, v in enumerate(cameras)}, pose)
            ctx = FusionContext(anchor_fit_inlier_mm=5, anchor_fit_min_anchors=6,
                triangulation_min_views=2, external_unit_scale_to_meter=1,
                max_triangulation_residual_mm=10, geometry_fit_residual_tau_mm=10,
                triangulation_weight_mode="linear", triangulation_weight_alpha=1,
                fit_weight_mode="uniform", geometry_fit_support_reference_views=4,
                geometry_fit_parallax_power=1, geometry_fit_parallax_stat="max",
                geometry_fit_min_weight=.01)
            tracker.predicted = coordinate
            fit = tracker._fusion_backend.solve(frame, ctx)
            assert fit.fit_status in {"ok", "no_inlier_refit"}, fit.fit_status
            np.testing.assert_allclose(fit.pose, pose, atol=2e-5)
            assert abs(fit.health["joint_value"] - coordinate) < 2e-5, fit.health
            # The joint has no measurements: keep a valid body solution and hold it.
            static = {i: rows for i, rows in observations.items() if j["part_ids"][i] == 0}
            held = tracker._fusion_backend.solve(FusionFrame(static, frame.intrinsics_map,
                                                             frame.extrinsics_map, pose), ctx)
            assert held.fit_status == "theta_unobserved", held.fit_status
            assert tracker._policy.decide("triangulate", held)[0]
            renderer = tracker._model.renderer
            renderer.set_object_joint_angle(1, coordinate)
            images = []
            for intr, extr in zip(K, cameras):
                camera = build_camera_model(384, 384, intr, extr @ pose)
                rgb = renderer.render_object_model(1, camera, [RenderType.COLOR])[RenderType.COLOR]
                images.append(np.clip(rgb * 255, 0, 255).astype(np.uint8))
            tracker.set_frame(0)
            first = tracker.step(images, K, cameras, pose)
            tracker.synchronize()
            tracker.set_frame(1)
            second = tracker.step(images, K, cameras, first.pose_T_w_o)
            tracker.synchronize()
            assert np.isfinite(second.pose_T_w_o).all()
            assert second.health["joint_type"] == kind
            assert np.isfinite(second.health["joint_value"])
            report[kind] = dict(perfect_observation_fit=fit.fit_status,
                held_joint_status=held.fit_status, gpu_status=second.status,
                gpu_fit_status=second.fit_status, gpu_joint_value=second.health["joint_value"],
                observer_compile=True, deploy_manifest=tracker.deploy_manifest,
                note="Plain generated RGB tests execution; pose accuracy assertions use perfect multiview observations")
            write_json(output / "validation.json", report)
        del tracker
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[validated] {kind}: {report[kind]['gpu_status']}", flush=True)


if __name__ == "__main__":
    main()
