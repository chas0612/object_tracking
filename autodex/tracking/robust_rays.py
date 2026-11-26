"""Opt-in deterministic camera-ray consensus; external world units throughout."""
import numpy as np


def consensus_point(centres, rays, weights, *, threshold, min_views=3,
                    min_angle_deg=10., min_weight_fraction=.55):
    c=np.asarray(centres,float);d=np.asarray(rays,float);w=np.asarray(weights,float)
    if len(c)<min_views or not np.isfinite(c).all() or not np.isfinite(d).all():return None
    if not np.isfinite(w).all() or w.sum()<=0 or np.any(np.linalg.norm(d,axis=1)<=1e-12):return None
    d=d/np.linalg.norm(d,axis=1)[:,None];w=np.maximum(w,1e-12)
    P=np.eye(3)[None]-d[:,:,None]*d[:,None,:]
    A=P*w[:,None,None];b=(A@c[:,:,None])[:,:,0]
    i,j=np.triu_indices(len(c),1)
    angle=np.degrees(np.arccos(np.clip(np.abs(np.sum(d[i]*d[j],axis=1)),0,1)))
    good=angle>=min_angle_deg
    if not good.any():return None
    i,j=i[good],j[good]
    mats=np.concatenate([A.sum(axis=0)[None],A[i]+A[j]])
    rhs=np.concatenate([b.sum(axis=0)[None],b[i]+b[j]])
    good=np.isfinite(mats).all(axis=(1,2)) & (np.linalg.cond(mats)<1e10)
    if not good.any():return None
    candidates=np.linalg.solve(mats[good],rhs[good,:,None])[:,:,0]
    off=candidates[:,None]-c[None];along=np.sum(off*d[None],axis=2)
    errors=np.linalg.norm(off-along[:,:,None]*d[None],axis=2)
    masks=(errors<=threshold)&(along>0)
    mass=masks@w;count=masks.sum(axis=1)
    usable=(count>=min_views)&(mass>=min_weight_fraction*w.sum())
    if not usable.any():return None
    cost=np.sum(np.minimum(errors,threshold)*w[None],axis=1)/w.sum()
    score=np.where(usable,mass-cost*1e-7/threshold,-np.inf)
    mask=masks[int(np.argmax(score))]
    for _ in range(4):
        matrix=A[mask].sum(axis=0)
        if np.linalg.cond(matrix)>1e10:return None
        point=np.linalg.solve(matrix,b[mask].sum(axis=0))
        off=point-c;along=np.sum(off*d,axis=1)
        err=np.linalg.norm(off-along[:,None]*d,axis=1)
        updated=(err<=threshold)&(along>0)
        if updated.sum()<min_views or w[updated].sum()<min_weight_fraction*w.sum():return None
        if np.array_equal(updated,mask):break
        mask=updated
    else:return None
    i,j=np.triu_indices(int(mask.sum()),1)
    angle=np.degrees(np.arccos(np.clip(np.sum(d[mask][i]*d[mask][j],axis=1),-1,1)))
    if np.minimum(angle,180-angle).max()<min_angle_deg:return None
    return dict(point=point,mask=mask,residual=float(np.average(err[mask],weights=w[mask])),
                min_ray_angle_deg=float(angle.min()),mean_ray_angle_deg=float(angle.mean()))


def triangulate_consensus(observations_by_anchor,intrinsics_map,extrinsics_map,*,
                          min_views,external_unit_scale_to_meter,threshold_mm=8.,
                          min_angle_deg=10.,min_weight_fraction=.55,
                          weight_mode='linear',weight_alpha=1.):
    from utils.multiview_geometry import transform_confidence_weights
    scale=float(external_unit_scale_to_meter);cameras={}
    for cid,K in intrinsics_map.items():
        E=np.asarray(extrinsics_map[cid],float)
        cameras[cid]=(np.linalg.inv(K),E[:3,:3].T,-E[:3,:3].T@E[:3,3])
    records=[];rejected={};input_rays=0;kept_rays=0;rejected_anchors=0
    for aid,observations in observations_by_anchor.items():
        obs=[o for o in observations if o.get('valid_flag',True)]
        input_rays+=len(obs);c=[];d=[];conf=[]
        for ob in obs:
            inv,Rt,centre=cameras[str(ob['camera_id'])]
            c.append(centre);d.append(Rt@inv@np.r_[ob['uv_curr'],1.]);conf.append(ob.get('confidence',1.))
        if len(obs)<min_views:answer=None
        else:
            weights=transform_confidence_weights(np.asarray(conf),mode=weight_mode,alpha=weight_alpha)
            answer=consensus_point(c,d,weights,threshold=threshold_mm/1000/scale,
                min_views=min_views,min_angle_deg=min_angle_deg,min_weight_fraction=min_weight_fraction)
        if answer is None:rejected_anchors+=1;mask=np.zeros(len(obs),bool)
        else:mask=answer['mask']
        for ob,keep in zip(obs,mask):
            if not keep:
                cid=str(ob['camera_id']);rejected[cid]=rejected.get(cid,0)+1
        if answer is None:continue
        kept_rays+=int(mask.sum());conf=np.asarray(conf);weights=np.asarray(weights);residual=answer['residual']
        records.append(dict(anchor_id=int(aid),position_o=np.asarray(obs[0]['position_o'],np.float32),
            position_w=answer['point'].astype(np.float32),num_support_views=int(mask.sum()),
            mean_confidence=float(weights[mask].mean()),mean_raw_confidence=float(conf[mask].mean()),
            triangulation_residual_external_units=residual,triangulation_residual_m=residual*scale,
            triangulation_residual_mm=residual*scale*1000,
            min_ray_angle_deg=answer['min_ray_angle_deg'],mean_ray_angle_deg=answer['mean_ray_angle_deg'],
            support_camera_ids=[str(ob['camera_id']) for ob,k in zip(obs,mask) if k]))
    return dict(status='ok' if records else 'no_triangulated_anchors',records=records,
        robust_ray_health=dict(consensus_input_rays=input_rays,consensus_kept_rays=kept_rays,
            consensus_rejected_anchors=rejected_anchors,consensus_rejected_by_view=rejected))
