"""Opt-in constrained articulated fit; the pinned runtime is never modified.

All public lengths are converted from metres through FusionContext. A rotation
prior is a soft covariance regularizer, not a forced orientation or ground truth.
Rejected fits return no candidate: upstream reports the real held/fallback pose.
"""
from pathlib import Path
from collections import deque

import numpy as np


def gate_view_rays(frame, joint, coordinate, max_distance):
    """Discard individual implausible rays, not a whole poisoned 3D anchor.

    Distance is in external world units. This is a local tracking search bound,
    not evidence that an observation is correct; robust fit/pose gates still run.
    """
    from utils.multiview_geometry import joint_transform
    camera = {}
    for cid, K in frame.intrinsics_map.items():
        E = np.asarray(frame.extrinsics_map[cid], float)
        camera[cid] = (np.linalg.inv(K), E[:3,:3].T, -E[:3,:3].T@E[:3,3])
    J = joint_transform(joint['joint_axis_o'], joint['joint_origin_o'], coordinate,
                        joint_type=joint['joint_type'])
    prior = np.asarray(frame.prior_T_wo, float)
    kept, removed, counts = {}, 0, {}
    for aid, observations in frame.observations_by_anchor.items():
        selected=[]
        for obs in observations:
            if not obs.get('valid_flag',True): continue
            cid=str(obs['camera_id']); inv, Rt, centre = camera[cid]
            x=np.asarray(obs['position_o'],float)
            if joint['part_ids'][int(aid)]==1:x=J[:3,:3]@x+J[:3,3]
            expected=prior[:3,:3]@x+prior[:3,3]
            ray=Rt@inv@np.r_[obs['uv_curr'],1.];ray/=np.linalg.norm(ray)
            offset=expected-centre
            error=np.linalg.norm(offset-ray*(offset@ray))
            if np.isfinite(error) and error<=max_distance:
                selected.append(obs)
            else:
                removed+=1;counts[cid]=counts.get(cid,0)+1
        if len(selected)>=2: kept[aid]=selected
    return kept,removed,counts


def rigid_fit(source, target, weights, prior_rotation, prior_radius):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 1e-12:
        raise ValueError("No positively weighted anchors")
    x = np.sum(weights[:, None] * source, axis=0) / total
    y = np.sum(weights[:, None] * target, axis=0) / total
    covariance = ((source-x)*weights[:, None]).T @ (target-y)
    # min sum w||RX-Y||^2 + sum(w)*radius^2 ||R-R_prior||_F^2
    covariance += total * prior_radius**2 * prior_rotation.T
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3); correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    pose = np.eye(4); pose[:3, :3] = vt.T @ correction @ u.T
    pose[:3, 3] = y-pose[:3, :3] @ x
    return pose


def constrained_fit(source, target, weights, parts, axis, origin, joint_type,
                    initial, limits, prior_rotation, prior_radius, iterations=40,
                    observe_joint=True, body_from_static=False):
    from utils.multiview_geometry import (
        joint_transform, fit_joint_angle_weighted, fit_joint_displacement_weighted)
    moving = parts == 1
    solve = fit_joint_angle_weighted if joint_type == 'revolute' else fit_joint_displacement_weighted
    q = float(np.clip(initial, *limits)) if limits is not None else float(initial)
    boundary = False
    def pose_at(value):
        swung = source.copy()
        if moving.any():
            j = joint_transform(axis, origin, value, joint_type=joint_type)
            swung[moving] = source[moving] @ j[:3, :3].T + j[:3, 3]
        keep = ~moving if body_from_static else np.ones(len(parts), bool)
        if not keep.any():
            raise ValueError('Static-body fit requires static anchors')
        return rigid_fit(swung[keep], target[keep], weights[keep], prior_rotation, prior_radius)
    pose = pose_at(q)
    for used in range(1, iterations+1):
        if not observe_joint or not moving.any():
            break
        local = (target[moving]-pose[:3, 3]) @ pose[:3, :3]
        solved = float(solve(source[moving], local, weights[moving], axis, origin))
        if joint_type == 'revolute':
            solved = q + np.arctan2(np.sin(solved-q), np.cos(solved-q))
        bounded = float(np.clip(solved, *limits)) if limits is not None else solved
        boundary |= abs(bounded-solved) > 1e-8
        delta = bounded-q; q = bounded
        # Fit body at the *bounded* coordinate, including the final iteration.
        pose = pose_at(q)
        if abs(delta) < 1e-7:
            break
    return pose, q, used, boundary


class StableArticulatedBackend:
    name = 'triangulate'

    def __init__(self, owner, config, diagnostic_dir=None):
        self.owner, self.config = owner, dict(config)
        self.diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir else None
        defaults = dict(rotation_prior_radius_m=.008, min_static_inliers=12,
                        min_total_inliers=20, min_inlier_fraction=.3,
                        max_inlier_rms_mm=12., max_body_rotation_deg=35.,
                        max_body_translation_m=.05, fit_inlier_mm=18.,
                        max_triangulation_residual_mm=15., ray_prior_distance_m=.035, iterations=60,
                        wrong_surface_margin_m=.02, part_residual_ratio=3., min_view_support=2,
                        body_from_static=False,body_flow_visible_only=False,
                        flow_consistency_gate=False,max_flow_correction_m=.008,
                        max_flow_correction_deg=10.,min_flow_inliers=60,
                        part_support_gate=False,min_static_support_ratio=.4,
                        min_moving_support_ratio=.4,support_reference_window=20,
                        support_reference_update_ratio=.8,
                        ray_confidence_gate=False,ray_min_confidence=.2,
                        robust_ray_consensus=False,consensus_ray_mm=8.,
                        consensus_min_angle_deg=10.,consensus_min_weight_fraction=.55)
        unknown = set(config)-set(defaults)
        if unknown: raise ValueError(f'Unknown stability options: {sorted(unknown)}')
        defaults.update(config); self.config = defaults
        switches={'body_from_static','body_flow_visible_only','flow_consistency_gate',
                  'part_support_gate','ray_confidence_gate','robust_ray_consensus'}
        if any(not isinstance(defaults[k],bool) for k in switches):
            raise ValueError('Body-fit/visibility switches must be boolean')
        if any(not np.isfinite(v) or v <= 0 for k,v in defaults.items() if k not in switches):
            raise ValueError('Stability parameters must be finite and positive')
        if self.config['min_inlier_fraction'] > 1: raise ValueError('Invalid inlier fraction')
        if defaults['ray_min_confidence']>1 or defaults['consensus_min_weight_fraction']>1:
            raise ValueError('Invalid ray confidence/support fraction')
        if defaults['consensus_min_angle_deg']>=90:raise ValueError('Invalid consensus parallax')
        ratios=('min_static_support_ratio','min_moving_support_ratio','support_reference_update_ratio')
        if any(defaults[k]>1 for k in ratios):raise ValueError('Support ratios must be in (0, 1]')
        window=defaults['support_reference_window']
        if isinstance(window,bool) or int(window)!=window:raise ValueError('Support window must be an integer')
        if defaults['support_reference_update_ratio']<max(defaults[k] for k in ratios[:2]):
            raise ValueError('Reference update ratio must not be below rejection ratios')
        self.reset_support_history()

    def reset_support_history(self):
        """Discard warmup/previous sequence evidence; reverse passes bootstrap anew."""
        self.support_history={k:deque(maxlen=int(self.config['support_reference_window']))
                              for k in ('static','moving')}
        self.support_last_frame=None

    def support_reference(self, part):
        values=self.support_history[part]
        return float(np.median(values)) if values else None

    def remember_support(self, static_count, moving_count, observe_joint):
        if not self.config['part_support_gate'] or self.support_last_frame==self.owner.current_frame:
            return
        self.support_last_frame=self.owner.current_frame
        for part,count in [('static',static_count),('moving',moving_count)]:
            if part=='moving' and not observe_joint:continue
            ref=self.support_reference(part)
            # Rejected measurements never reach here. Weak-but-accepted support
            # must not immediately teach the history that occlusion is normal.
            if ref is None or count>=self.config['support_reference_update_ratio']*ref:
                self.support_history[part].append(int(count))

    def commit_support(self, result_status, health):
        """Called by the bridge *after* upstream policy/optional wrappers accept.

        A usable fusion candidate alone is not an accepted tracking result.
        """
        if result_status!='ok':
            health['body_observed']=False
            return
        if health.get('body_observed',False):
            self.remember_support(health['num_static_inliers'],health['num_moving_inliers'],
                                  bool(health.get('theta_observed',False)))

    def solve(self, frame, ctx):
        from multiview_solver.fusion_backends import FusionResult
        from utils.multiview_geometry import (triangulate_anchor_observations,
            build_fit_weights_from_triangulation_records, compute_articulated_fit_residuals,
            reject_wrong_surface_anchors)
        owner, cfg, joint = self.owner, self.config, self.owner.joint
        scale = float(ctx.external_unit_scale_to_meter)
        observations=frame.observations_by_anchor
        confidence_removed={}
        if cfg['ray_confidence_gate']:
            observations={}
            for aid,obs in frame.observations_by_anchor.items():
                selected=[]
                for ob in obs:
                    if np.isfinite(ob.get('confidence',0)) and ob.get('confidence',0)>=cfg['ray_min_confidence']:
                        selected.append(ob)
                    else:
                        cid=str(ob['camera_id']);confidence_removed[cid]=confidence_removed.get(cid,0)+1
                if selected:observations[aid]=selected
        removed, by_view = 0, {}
        if frame.intrinsics_map:
            from types import SimpleNamespace
            ray_frame=SimpleNamespace(observations_by_anchor=observations,
                intrinsics_map=frame.intrinsics_map,extrinsics_map=frame.extrinsics_map,prior_T_wo=frame.prior_T_wo)
            observations,removed,by_view=gate_view_rays(ray_frame,joint,float(owner.predicted),
                cfg['ray_prior_distance_m']/scale)
        triangulate=triangulate_anchor_observations;extra={}
        if cfg['robust_ray_consensus']:
            from autodex.tracking.robust_rays import triangulate_consensus
            triangulate=triangulate_consensus
            extra=dict(threshold_mm=cfg['consensus_ray_mm'],min_angle_deg=cfg['consensus_min_angle_deg'],
                       min_weight_fraction=cfg['consensus_min_weight_fraction'])
        tri = triangulate(observations_by_anchor=observations,
            intrinsics_map=frame.intrinsics_map, extrinsics_map=frame.extrinsics_map,
            min_views=max(ctx.triangulation_min_views,int(cfg['min_view_support'])), external_unit_scale_to_meter=scale,
            weight_mode=ctx.triangulation_weight_mode, weight_alpha=ctx.triangulation_weight_alpha,**extra)
        rows = [r for r in tri.get('records', [])
                if r['triangulation_residual_mm'] <= cfg['max_triangulation_residual_mm']]
        health = dict(tri_status=tri['status'], stability_enabled=True,
            num_triangulated=len(rows), num_input_anchors=len(rows), theta_observed=False,
            theta_rejected=True, theta_clamped=False, body_observed=False,joint_value=float(owner.coordinate),
            joint_type=joint['joint_type'])
        health.update(ray_prior_rejected=removed,ray_prior_rejected_by_view=by_view)
        health.update(ray_confidence_rejected_by_view=confidence_removed,
            ray_confidence_rejected=sum(confidence_removed.values()),robust_ray_consensus=cfg['robust_ray_consensus'])
        health.update(tri.get('robust_ray_health',{}))
        refs={part:self.support_reference(part) for part in ('static','moving')}
        health.update(part_support_gate=cfg['part_support_gate'],
            static_support_reference=refs['static'],moving_support_reference=refs['moving'])
        def reject(reason):
            health.update(status='stability_rejected', stability_rejection_reason=reason)
            return FusionResult(None, 'stability_rejected', 0., health)
        if len(rows) < cfg['min_total_inliers']: return reject('insufficient_triangulated')
        source = np.asarray([r['position_o'] for r in rows], float)
        target = np.asarray([r['position_w'] for r in rows], float)
        ids = np.asarray([r['anchor_id'] for r in rows], int)
        parts = joint['part_ids'][ids]
        support_views=np.asarray([r.get('num_support_views',np.nan) for r in rows],float)
        parallax=np.asarray([r.get('mean_ray_angle_deg',np.nan) for r in rows],float)
        weights = build_fit_weights_from_triangulation_records(rows, mode=ctx.fit_weight_mode,
            geometry_support_reference_views=ctx.geometry_fit_support_reference_views,
            geometry_residual_tau_mm=ctx.geometry_fit_residual_tau_mm,
            geometry_parallax_power=ctx.geometry_fit_parallax_power,
            geometry_parallax_stat=ctx.geometry_fit_parallax_stat,
            geometry_min_weight=ctx.geometry_fit_min_weight).astype(float)
        # The bounded backend must not bypass upstream's part-identity gate.
        # A static anchor tracked onto the other arm is not a body measurement.
        if 'part_surface_samples_o' in joint:
            rejection = reject_wrong_surface_anchors(target,parts,source_points_o=source,
                pose_world_from_object=frame.prior_T_wo,theta=float(owner.predicted),
                axis_o=joint['joint_axis_o'],origin_o=joint['joint_origin_o'],
                part_surface_samples_o=joint['part_surface_samples_o'],
                part_surface_offsets=joint['part_surface_offsets'],
                margin_external=cfg['wrong_surface_margin_m']/scale,joint_type=joint['joint_type'])
            keep=rejection['keep_mask']
            health['wrong_surface_rejected']=int(rejection['num_rejected'])
            source,target,parts,weights=source[keep],target[keep],parts[keep],weights[keep]
            ids,support_views,parallax=ids[keep],support_views[keep],parallax[keep]
            health['num_surface_kept']=int(sum(keep))
            if len(source)<cfg['min_total_inliers']:return reject('insufficient_part_support')
        static = parts == 0
        health['num_moving_anchors'] = int(sum(~static))
        if sum(static) < cfg['min_static_inliers']: return reject('insufficient_static')
        observe = not owner.pinned and sum(~static) >= owner.min_moving_anchors
        initial = float(owner.predicted)
        prior = np.asarray(frame.prior_T_wo, float)
        limits = joint['joint_limits']
        axis, origin = joint['joint_axis_o'], joint['joint_origin_o']
        cutoff = cfg['fit_inlier_mm']/1000/scale
        selected = np.ones(len(source), bool)
        pose, q, clipped, rounds = prior.copy(), initial, False, 0
        # Tukey-style trimming/refit, with the same prior and hard joint bounds.
        for _ in range(3):
            if selected.sum() < cfg['min_total_inliers'] or sum(selected & static) < cfg['min_static_inliers']:
                return reject('insufficient_inlier_support')
            pose, q, rounds, bounded = constrained_fit(source[selected], target[selected], weights[selected],
                parts[selected], axis, origin, joint['joint_type'], initial, limits,
                prior[:3, :3], cfg['rotation_prior_radius_m']/scale,
                int(cfg['iterations']), observe_joint=observe,body_from_static=cfg['body_from_static'])
            clipped |= bounded
            residual = compute_articulated_fit_residuals(pose, q, source, target, parts,
                axis, origin, joint_type=joint['joint_type'])
            updated = residual <= cutoff
            if np.array_equal(updated, selected): break
            selected = updated
        # Refit the final selected set; recompute masks at the returned pose/q.
        if selected.sum() < cfg['min_total_inliers'] or sum(selected & static) < cfg['min_static_inliers']:
            return reject('insufficient_inlier_support')
        if observe and sum(selected & ~static) < owner.min_moving_anchors:
            observe = False; initial = float(owner.coordinate); selected &= static
            if selected.sum() < cfg['min_static_inliers']: return reject('insufficient_static')
        pose, q, rounds, bounded = constrained_fit(source[selected], target[selected], weights[selected],
            parts[selected], axis, origin, joint['joint_type'], initial, limits,
            prior[:3, :3], cfg['rotation_prior_radius_m']/scale,
            int(cfg['iterations']), observe_joint=observe,body_from_static=cfg['body_from_static'])
        clipped |= bounded
        residual = compute_articulated_fit_residuals(pose, q, source, target, parts,
            axis, origin, joint_type=joint['joint_type'])
        static_median=float(np.median(residual[static]))
        moving_median=float(np.median(residual[~static])) if (~static).any() else 0.
        health.update(candidate_static_median_mm=static_median*scale*1000,
            candidate_moving_median_mm=moving_median*scale*1000,body_from_static=cfg['body_from_static'])
        candidate_inliers=residual<=cutoff
        candidate_moving=int(sum(candidate_inliers & ~static))
        health.update(candidate_static_inliers=int(sum(candidate_inliers & static)),
                      candidate_moving_inliers=candidate_moving)
        moving_weak=bool(observe and cfg['part_support_gate'] and refs['moving'] is not None
                         and candidate_moving<cfg['min_moving_support_ratio']*refs['moving'])
        health['moving_support_retention']=(candidate_moving/refs['moving'] if refs['moving'] else None)
        # Restore the original part imbalance check, before a moving-arm trim can
        # hide a bad joint fit. Hold q and refit only the body; do not call it a
        # successful joint measurement.
        imbalance=observe and moving_median>max(cfg['part_residual_ratio']*static_median,.5*cutoff)
        joint_rejected=imbalance or moving_weak
        if joint_rejected:
            health['joint_rejection_reason']='moving_support_drop' if moving_weak else 'moving_static_residual_imbalance'
            observe=False;q=float(owner.coordinate)
            body_selected=selected & static
            pose=rigid_fit(source[body_selected],target[body_selected],weights[body_selected],prior[:3,:3],
                cfg['rotation_prior_radius_m']/scale)
            residual=compute_articulated_fit_residuals(pose,q,source,target,parts,axis,origin,
                joint_type=joint['joint_type'])
        inliers = residual <= cutoff
        if joint_rejected:inliers &= static
        health.update(static_median_mm=float(np.median(residual[static]))*scale*1000,
            moving_median_mm=float(np.median(residual[~static]))*scale*1000 if (~static).any() else 0.)
        rms = float(np.sqrt(np.mean(residual[inliers]**2))*scale*1000) if inliers.any() else float('inf')
        rotation = float(np.degrees(np.arccos(np.clip((np.trace(prior[:3, :3].T@pose[:3, :3])-1)/2,-1,1))))
        translation = float(np.linalg.norm(prior[:3, 3]-pose[:3, 3])*scale)
        health.update(num_inlier_anchors=int(sum(inliers)), num_static_inliers=int(sum(inliers & static)),
            num_moving_inliers=int(sum(inliers & ~static)), inlier_rms_mm=rms,
            mean_residual_mm=float(residual.mean()*scale*1000),
            max_residual_mm=float(residual.max()*scale*1000), num_rounds=rounds,
            body_rotation_step_deg=rotation, body_translation_step_m=translation,
            bounded_joint_candidate=float(q), bounded_joint_hit=clipped)
        static_count=int(sum(inliers & static))
        health['static_support_retention']=(static_count/refs['static'] if refs['static'] else None)
        health['body_observed']=False
        for label,mask in [('static',inliers & static),('moving',inliers & ~static)]:
            for name,values in [('support_views',support_views),('parallax_deg',parallax)]:
                valid=values[mask & np.isfinite(values)]
                if len(valid):
                    health[f'{label}_inlier_{name}_median']=float(np.median(valid))
                    health[f'{label}_inlier_{name}_min']=float(valid.min())
        reason = None
        if sum(inliers & static) < cfg['min_static_inliers']: reason='insufficient_static_inliers'
        elif (cfg['part_support_gate'] and refs['static'] is not None
              and static_count<cfg['min_static_support_ratio']*refs['static']): reason='static_support_drop'
        elif inliers.sum() < cfg['min_total_inliers']: reason='insufficient_total_inliers'
        elif inliers.mean() < cfg['min_inlier_fraction']: reason='low_inlier_fraction'
        elif rms > cfg['max_inlier_rms_mm']: reason='large_inlier_rms'
        elif rotation > cfg['max_body_rotation_deg']: reason='large_body_rotation'
        elif translation > cfg['max_body_translation_m']: reason='large_body_translation'
        elif not np.isfinite(pose).all(): reason='nonfinite_pose'
        flow=getattr(owner,'flow_prior_health',{})
        flow_constraint=bool(cfg['flow_consistency_gate'] and flow.get('body_flow_bootstrap')
            and flow.get('body_flow_inliers',0)>=cfg['min_flow_inliers'])
        health['flow_body_consistency_gate']=flow_constraint
        if reason is None and flow_constraint:
            if translation>cfg['max_flow_correction_m']:reason='flow_body_translation_disagreement'
            elif rotation>cfg['max_flow_correction_deg']:reason='flow_body_rotation_disagreement'
        if self.diagnostic_dir and (reason or joint_rejected):
            self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.diagnostic_dir/f'frame_{owner.current_frame:06d}.npz',
                source=source,target=target,weights=weights,part_ids=parts,
                anchor_ids=ids,support_views=support_views,mean_ray_angle_deg=parallax,
                pose=pose,prior=prior,joint_value=q,residuals_m=residual*scale,inliers=inliers)
        if reason: return reject(reason)
        health.update(status='ok' if observe or owner.pinned else ('theta_rejected' if joint_rejected else 'theta_unobserved'),
            body_observed=True,
            joint_value=float(q),theta_observed=bool(observe or owner.pinned),
            theta_rejected=not bool(observe or owner.pinned),theta_clamped=bool(clipped and (observe or owner.pinned)),
            joint_source='external_anchor' if owner.pinned else 'bounded_stability')
        if joint['joint_type']=='revolute': health.update(theta_rad=q,theta_deg=float(np.degrees(q)))
        return FusionResult(pose.astype(np.float32), health['status'], float(inliers.mean()), health,
                            'articulated_bounded_stability')
