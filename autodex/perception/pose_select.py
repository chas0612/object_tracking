"""Cross-view pose selection utilities.

Given multiple candidate object poses (each from a different source view) and
ground-truth masks across all cameras, pick the candidate whose rendered
silhouette best matches the masks (mean IoU across views).

Used by:
  - PerceptionPipeline._select_best_pose (FoundationPose register candidates)
  - FoundPose / PicoPose first-frame init scripts (per-view PnP candidates)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


_FP_ROOT = Path(__file__).resolve().parent / "thirdparty/FoundationPose"
if str(_FP_ROOT) not in sys.path:
    sys.path.insert(0, str(_FP_ROOT))


def _render_silhouette(
    pose_world: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
    H: int,
    W: int,
    glctx,
    mesh_tensors,
) -> np.ndarray:
    """Render mesh silhouette in target camera given world pose. Returns bool (H, W)."""
    from Utils import nvdiffrast_render

    pose_cam = extrinsic @ pose_world
    pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
    K = np.asarray(K, dtype=np.float32)
    rc, _, _ = nvdiffrast_render(
        K=K, H=H, W=W, ob_in_cams=pt, glctx=glctx,
        mesh_tensors=mesh_tensors, use_light=False,
    )
    return rc[0].detach().cpu().numpy().sum(axis=2) > 0


def _build_view_batch(
    masks: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    extrinsics: Dict[str, np.ndarray],
    H: int,
    W: int,
    device: str = "cuda",
):
    """Stack per-view tensors used by batched IoU rendering.

    Returns (serials, mask_batch (N,H,W) bool, extrinsic_batch (N,4,4),
             proj_batch (N,4,4), glcam_t (4,4)) or None if no valid views.
    """
    from Utils import projection_matrix_from_intrinsics, glcam_in_cvcam

    serials: list = []
    mask_list: list = []
    extr_list: list = []
    proj_list: list = []
    for s, mask in masks.items():
        if s not in intrinsics or s not in extrinsics:
            continue
        K = np.asarray(intrinsics[s], dtype=np.float32)
        proj = projection_matrix_from_intrinsics(K, height=H, width=W,
                                                  znear=0.001, zfar=100)
        proj_list.append(torch.as_tensor(proj.reshape(4, 4), device=device,
                                          dtype=torch.float32))
        extr_list.append(torch.as_tensor(np.asarray(extrinsics[s]),
                                          device=device, dtype=torch.float32))
        mask_list.append(torch.as_tensor(mask, device=device, dtype=torch.bool))
        serials.append(s)

    if not serials:
        return None

    mask_batch = torch.stack(mask_list, dim=0)
    extrinsic_batch = torch.stack(extr_list, dim=0)
    proj_batch = torch.stack(proj_list, dim=0)
    glcam_t = torch.as_tensor(glcam_in_cvcam, device=device, dtype=torch.float32)
    return serials, mask_batch, extrinsic_batch, proj_batch, glcam_t


def _render_silhouette_bool_batched(
    H: int, W: int,
    ob_in_cams: torch.Tensor,   # (N, 4, 4)
    proj_t: torch.Tensor,       # (N, 4, 4)
    glcam_t: torch.Tensor,      # (4, 4)
    glctx,
    mesh_tensors,
    pos_homo: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched silhouette mask render. Returns bool (N, H, W)."""
    import nvdiffrast.torch as dr
    from Utils import to_homo_torch

    pos = mesh_tensors["pos"]
    faces = mesh_tensors["faces"]
    pos_homo = to_homo_torch(pos) if pos_homo is None else pos_homo
    ob_in_glcams = glcam_t[None] @ ob_in_cams                     # (N, 4, 4)
    proj_pose = proj_t @ ob_in_glcams                              # (N, 4, 4)
    pos_clip = proj_pose[:, None] @ pos_homo[None, ..., None]      # (N, V, 4, 1)
    pos_clip = pos_clip[..., 0]                                    # (N, V, 4)
    rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=np.asarray([H, W]))
    sil = rast_out[..., -1] > 0                                    # (N, H, W)
    sil = torch.flip(sil, dims=[1])
    return sil


def compute_cross_view_iou(
    pose_world: np.ndarray,
    masks: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    extrinsics: Dict[str, np.ndarray],
    H: int,
    W: int,
    glctx,
    mesh_tensors,
) -> Tuple[float, Dict[str, float]]:
    """Mean IoU of rendered mesh vs SAM mask across all views (batched).

    Stacks all valid views into a single rasterize call instead of looping.
    Returns mean IoU and per-view IoU (serial -> float).
    """
    batch = _build_view_batch(masks, intrinsics, extrinsics, H, W)
    if batch is None:
        return 0.0, {}
    serials, mask_batch, extrinsic_batch, proj_batch, glcam_t = batch

    pose_world_t = torch.as_tensor(pose_world, device="cuda", dtype=torch.float32)
    pose_cam_batch = extrinsic_batch @ pose_world_t                # (N, 4, 4)
    sil = _render_silhouette_bool_batched(
        H=H, W=W, ob_in_cams=pose_cam_batch, proj_t=proj_batch,
        glcam_t=glcam_t, glctx=glctx, mesh_tensors=mesh_tensors,
    )                                                              # (N, H, W) bool

    inter = (sil & mask_batch).sum(dim=(1, 2)).float()
    union = (sil | mask_batch).sum(dim=(1, 2)).float()
    per_view_iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))

    iou_arr = per_view_iou.cpu().tolist()
    per_view = dict(zip(serials, iou_arr))
    return float(np.mean(iou_arr)) if iou_arr else 0.0, per_view


def select_best_pose_by_iou(
    candidates: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    extrinsics: Dict[str, np.ndarray],
    H: int,
    W: int,
    glctx,
    mesh_tensors,
) -> Tuple[Optional[str], Optional[np.ndarray], float, Dict[str, float]]:
    """Pick candidate pose with highest mean cross-view mask IoU.

    Args:
        candidates: source_serial -> 4x4 pose_world. Each entry is a candidate.
        masks: serial -> bool (H, W) mask (used as ground truth for IoU).
        intrinsics, extrinsics, H, W, glctx, mesh_tensors: see compute_cross_view_iou.

    Returns:
        best_serial, best_pose_world, best_mean_iou, per_candidate_mean_iou (serial -> float).
        Returns (None, None, -1, {}) if no candidate produced any IoU.
    """
    best_serial: Optional[str] = None
    best_pose: Optional[np.ndarray] = None
    best_iou: float = -1.0
    per_cand: Dict[str, float] = {}

    for src_s, pose_world in candidates.items():
        mean_iou, _ = compute_cross_view_iou(
            pose_world, masks, intrinsics, extrinsics, H, W, glctx, mesh_tensors,
        )
        per_cand[src_s] = mean_iou
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_serial = src_s
            best_pose = pose_world

    return best_serial, best_pose, best_iou, per_cand


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 0:
        raise ValueError("Quaternion norm must be positive")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def sample_so3_rotations(count: int) -> list[np.ndarray]:
    """Return deterministic low-discrepancy rotations covering SO(3).

    This is the Shoemake uniform-quaternion construction driven by irrational
    additive sequences.  It avoids a scipy dependency and, unlike Euler-angle
    grids, does not oversample the poles.
    """
    if count < 1:
        raise ValueError("count must be positive")
    phi_inv = (np.sqrt(5.0) - 1.0) * 0.5
    sqrt2_frac = np.sqrt(2.0) - 1.0
    rotations: list[np.ndarray] = []
    for index in range(count):
        u1 = (index + 0.5) / count
        u2 = ((index + 0.5) * phi_inv) % 1.0
        u3 = ((index + 0.5) * sqrt2_frac) % 1.0
        quaternion = np.asarray([
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ])
        rotations.append(_quaternion_xyzw_to_matrix(quaternion))
    return rotations


def _translation_medoid(candidates: Mapping[str, np.ndarray]) -> np.ndarray:
    translations = np.stack([
        np.asarray(pose, dtype=np.float64)[:3, 3] for pose in candidates.values()
    ])
    distances = np.linalg.norm(translations[:, None, :] - translations[None, :, :], axis=2)
    return translations[int(np.argmin(distances.sum(axis=1)))].copy()


def build_global_pose_hypotheses(
    candidates: Mapping[str, np.ndarray],
    rotation_count: int = 256,
    translation_candidates: Mapping[str, np.ndarray] | None = None,
    candidate_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build coarse global pose hypotheses around a robust translation.

    The pool contains every raw per-view FoundPose pose, each FoundPose
    rotation recombined with the translation medoid, and a deterministic SO(3)
    covering at that medoid.  Recombination is important for cases where one
    camera has the right rotation while a different camera has the right
    translation.
    """
    if not candidates:
        return []
    translation = _translation_medoid(translation_candidates or candidates)
    hypotheses: list[dict[str, Any]] = []
    for serial, value in candidates.items():
        pose = np.asarray(value, dtype=np.float64)
        metadata = dict((candidate_metadata or {}).get(serial, {}))
        hypotheses.append({"source": f"foundpose:{serial}", "pose_world": pose.copy(), **metadata})
        recombined = pose.copy()
        recombined[:3, 3] = translation
        hypotheses.append({"source": f"foundpose_rotation:{serial}", "pose_world": recombined, **metadata})
    for index, rotation in enumerate(sample_so3_rotations(rotation_count)):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        hypotheses.append({"source": f"so3:{index:04d}", "pose_world": pose})
    return hypotheses


def _resize_view_inputs(
    masks: Mapping[str, np.ndarray],
    intrinsics: Mapping[str, np.ndarray],
    max_side: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, int]:
    if max_side < 32:
        raise ValueError("max_side must be at least 32")
    first = next(iter(masks.values()))
    height, width = np.asarray(first).shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    resized_masks: dict[str, np.ndarray] = {}
    resized_intrinsics: dict[str, np.ndarray] = {}
    for serial, mask in masks.items():
        value = np.asarray(mask, dtype=np.uint8)
        if value.shape[:2] != (height, width):
            continue
        resized_masks[serial] = cv2.resize(
            value, (out_width, out_height), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if serial in intrinsics:
            matrix = np.asarray(intrinsics[serial], dtype=np.float64).copy()
            matrix[0, :] *= out_width / width
            matrix[1, :] *= out_height / height
            resized_intrinsics[serial] = matrix
    return resized_masks, resized_intrinsics, out_height, out_width


def score_pose_hypotheses_by_iou(
    hypotheses: Sequence[Mapping[str, Any]],
    masks: Mapping[str, np.ndarray],
    intrinsics: Mapping[str, np.ndarray],
    extrinsics: Mapping[str, np.ndarray],
    glctx,
    mesh_tensors,
    max_side: int = 160,
    trim_fraction: float = 0.15,
) -> list[dict[str, Any]]:
    """Coarsely rank pose hypotheses with robust cross-view silhouette IoU."""
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    resized_masks, resized_intrinsics, height, width = _resize_view_inputs(
        masks, intrinsics, max_side,
    )
    batch = _build_view_batch(
        dict(resized_masks), dict(resized_intrinsics), dict(extrinsics), height, width,
    )
    if batch is None:
        return []
    from Utils import to_homo_torch
    _, mask_batch, extrinsic_batch, proj_batch, glcam_t = batch
    pos_homo = to_homo_torch(mesh_tensors["pos"])
    ranked: list[dict[str, Any]] = []
    with torch.no_grad():
        for hypothesis in hypotheses:
            pose = torch.as_tensor(hypothesis["pose_world"], device="cuda", dtype=torch.float32)
            sil = _render_silhouette_bool_batched(
                H=height, W=width,
                ob_in_cams=extrinsic_batch @ pose,
                proj_t=proj_batch, glcam_t=glcam_t,
                glctx=glctx, mesh_tensors=mesh_tensors, pos_homo=pos_homo,
            )
            inter = (sil & mask_batch).sum(dim=(1, 2)).float()
            union = (sil | mask_batch).sum(dim=(1, 2)).float()
            values = torch.where(union > 0, inter / union, torch.zeros_like(inter)).cpu().numpy()
            values = np.sort(values.astype(np.float64, copy=False))
            mean_iou = float(values.mean()) if len(values) else 0.0
            trim = int(np.floor(len(values) * trim_fraction))
            robust_values = values[trim:len(values) - trim] if trim > 0 else values
            robust_iou = float(robust_values.mean()) if len(robust_values) else mean_iou
            ranked.append({
                **hypothesis,
                "coarse_mean_iou": float(mean_iou),
                "coarse_robust_iou": robust_iou,
            })
    return sorted(ranked, key=lambda item: item["coarse_robust_iou"], reverse=True)


def select_diverse_pose_hypotheses(
    ranked: Sequence[Mapping[str, Any]],
    count: int,
    min_rotation_deg: float,
    min_translation_m: float,
) -> list[dict[str, Any]]:
    """Greedily keep high-scoring hypotheses from distinct pose basins."""
    if count < 1 or min_rotation_deg < 0 or min_translation_m < 0:
        raise ValueError("diversity parameters must be non-negative and count positive")
    selected: list[dict[str, Any]] = []
    for item in ranked:
        pose = np.asarray(item["pose_world"], dtype=np.float64)
        duplicate = False
        for kept in selected:
            other = np.asarray(kept["pose_world"], dtype=np.float64)
            rotation = _rotation_distance_deg(pose, other)
            translation = float(np.linalg.norm(pose[:3, 3] - other[:3, 3]))
            if rotation < min_rotation_deg and translation < min_translation_m:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(dict(item))
        if len(selected) >= count:
            break
    return selected


def has_rotation_ambiguous_top_poses(
    ranked: Sequence[Mapping[str, Any]],
    score_key: str = "robust_iou",
    score_margin: float = 0.005,
    min_rotation_deg: float = 25.0,
) -> bool:
    """Return true when similarly scored poses disagree substantially in rotation."""
    if len(ranked) < 2 or score_margin < 0 or min_rotation_deg < 0:
        return False
    ordered = sorted(ranked, key=lambda item: float(item[score_key]), reverse=True)
    best = ordered[0]
    best_score = float(best[score_key])
    best_pose = np.asarray(best["pose_world"], dtype=np.float64)
    return any(
        best_score - float(item[score_key]) <= score_margin
        and _rotation_distance_deg(best_pose, np.asarray(item["pose_world"], dtype=np.float64)) >= min_rotation_deg
        for item in ordered[1:]
    )


def score_pose_hypotheses_by_asymmetry(
    hypotheses: Sequence[Mapping[str, Any]],
    masks: Mapping[str, np.ndarray],
    intrinsics: Mapping[str, np.ndarray],
    extrinsics: Mapping[str, np.ndarray],
    glctx,
    mesh_tensors,
    max_side: int = 512,
    disagreement_dilation: int = 9,
    asymmetry_weight: float = 0.7,
    trim_fraction: float = 0.15,
    min_disagreement_pixels: int = 16,
) -> list[dict[str, Any]]:
    """Rerank refined poses using only silhouette regions that distinguish them.

    Large near-symmetric bodies dominate ordinary IoU.  The union-minus-
    intersection of candidate renders isolates small symmetry-breaking geometry
    such as a loop or handle.  Candidate error in that automatically derived
    region is combined with its existing robust full-silhouette IoU.
    """
    if len(hypotheses) < 2:
        return [dict(item) for item in hypotheses]
    if not 0.0 <= asymmetry_weight <= 1.0 or not 0.0 <= trim_fraction < 0.5:
        raise ValueError("asymmetry_weight and trim_fraction are out of range")
    if disagreement_dilation < 1 or min_disagreement_pixels < 1:
        raise ValueError("disagreement parameters must be positive")
    if disagreement_dilation % 2 == 0:
        disagreement_dilation += 1

    resized_masks, resized_intrinsics, height, width = _resize_view_inputs(
        masks, intrinsics, max_side,
    )
    batch = _build_view_batch(
        dict(resized_masks), dict(resized_intrinsics), dict(extrinsics), height, width,
    )
    if batch is None:
        return [dict(item) for item in hypotheses]
    from Utils import to_homo_torch
    _, mask_batch, extrinsic_batch, proj_batch, glcam_t = batch
    pos_homo = to_homo_torch(mesh_tensors["pos"])
    rendered: list[torch.Tensor] = []
    with torch.no_grad():
        for item in hypotheses:
            pose = torch.as_tensor(item["pose_world"], device="cuda", dtype=torch.float32)
            rendered.append(_render_silhouette_bool_batched(
                H=height, W=width,
                ob_in_cams=extrinsic_batch @ pose,
                proj_t=proj_batch, glcam_t=glcam_t,
                glctx=glctx, mesh_tensors=mesh_tensors, pos_homo=pos_homo,
            ))
        render_stack = torch.stack(rendered, dim=0)  # (K,N,H,W)
        disagreement = render_stack.any(dim=0) & ~render_stack.all(dim=0)
        region = torch.nn.functional.max_pool2d(
            disagreement.float().unsqueeze(1),
            kernel_size=disagreement_dilation,
            stride=1,
            padding=disagreement_dilation // 2,
        ).squeeze(1) > 0
        region_pixels = region.sum(dim=(1, 2))
        valid_views = region_pixels >= min_disagreement_pixels

        reranked: list[dict[str, Any]] = []
        for item, silhouette in zip(hypotheses, rendered):
            mismatch = (silhouette ^ mask_batch) & region
            per_view_score = 1.0 - mismatch.sum(dim=(1, 2)).float() / region_pixels.clamp_min(1).float()
            values = per_view_score[valid_views].cpu().numpy().astype(np.float64, copy=False)
            values.sort()
            trim = int(np.floor(len(values) * trim_fraction))
            kept = values[trim:len(values) - trim] if trim > 0 else values
            asymmetry_score = float(kept.mean()) if len(kept) else 0.0
            full_score = float(item.get("robust_iou", item.get("mean_iou", 0.0)))
            combined = (1.0 - asymmetry_weight) * full_score + asymmetry_weight * asymmetry_score
            reranked.append({
                **item,
                "asymmetry_score": asymmetry_score,
                "asymmetry_combined_score": combined,
                "asymmetry_valid_views": int(valid_views.sum().item()),
                "asymmetry_mean_region_pixels": float(region_pixels[valid_views].float().mean().item()) if valid_views.any() else 0.0,
            })
    return sorted(reranked, key=lambda item: item["asymmetry_combined_score"], reverse=True)


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Geodesic SO(3) distance, with clipping for numerical stability."""
    relative = np.asarray(first, dtype=np.float64)[:3, :3].T @ np.asarray(second, dtype=np.float64)[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rank_pose_candidates(
    *,
    candidates: Mapping[str, np.ndarray],
    pnp_quality: Mapping[str, float],
    iou_scores: Mapping[str, float],
    consensus_rotation_deg: float = 25.0,
    consensus_translation_m: float = 0.06,
) -> list[dict[str, Any]]:
    """Rank existing FoundPose per-view candidates without rerunning FoundPose.

    The legacy selector sees only rendered silhouette IoU, which cannot
    distinguish rotations that share a silhouette.  This helper retains that
    score and augments it with FoundPose's own PnP quality plus agreement among
    independent camera estimates.  It intentionally does not claim to resolve
    a true visual symmetry: low score margins are surfaced as ambiguity.
    """
    if consensus_rotation_deg <= 0 or consensus_translation_m <= 0:
        raise ValueError("Consensus tolerances must be positive")
    names = list(candidates)
    if not names:
        return []
    qualities = np.asarray([max(0.0, float(pnp_quality.get(name, 0.0))) for name in names])
    quality_scale = float(qualities.max()) if float(qualities.max()) > 0 else 1.0
    ious = np.asarray([float(iou_scores.get(name, 0.0)) for name in names])
    iou_min, iou_max = float(ious.min()), float(ious.max())
    iou_scale = max(iou_max - iou_min, 1e-8)

    ranked: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        pose = np.asarray(candidates[name], dtype=np.float64)
        agreement = 0.0
        total_weight = 0.0
        for other_name in names:
            other = np.asarray(candidates[other_name], dtype=np.float64)
            rotation = _rotation_distance_deg(pose, other)
            translation = float(np.linalg.norm(pose[:3, 3] - other[:3, 3]))
            kernel = float(np.exp(-0.5 * ((rotation / consensus_rotation_deg) ** 2 +
                                          (translation / consensus_translation_m) ** 2)))
            weight = max(0.01, float(pnp_quality.get(other_name, 0.0)))
            agreement += weight * kernel
            total_weight += weight
        consensus = agreement / total_weight if total_weight else 0.0
        iou = float(iou_scores.get(name, 0.0))
        iou_normalized = (iou - iou_min) / iou_scale if iou_max > iou_min else 1.0
        quality_normalized = float(qualities[index] / quality_scale)
        hybrid = 0.45 * iou_normalized + 0.25 * quality_normalized + 0.30 * consensus
        ranked.append({
            "source_serial": name,
            "pose_world": pose,
            "mean_iou": iou,
            "pnp_quality": float(pnp_quality.get(name, 0.0)),
            "consensus": consensus,
            "hybrid_score": hybrid,
        })
    return ranked


def load_masks_bool(
    mask_dir: Path,
    serials: List[str],
    threshold: int = 127,
) -> Dict[str, np.ndarray]:
    """Load uint8 masks from {mask_dir}/{serial}.png as bool (H, W) dict.

    Skips files that don't exist or fail to load.
    """
    out: Dict[str, np.ndarray] = {}
    for s in serials:
        mp = Path(mask_dir) / f"{s}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        out[s] = m > threshold
    return out
