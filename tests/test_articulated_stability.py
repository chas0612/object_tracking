"""CPU tests against actual pinned upstream geometry; no GPU/model required."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'autodex/perception/thirdparty/MV-GoTrack-next')]
from autodex.tracking.articulated_stability import constrained_fit, rigid_fit, StableArticulatedBackend, gate_view_rays
from utils.multiview_geometry import joint_transform
from multiview_solver.fusion_backends import FusionFrame, FusionContext

def main():
    rng=np.random.default_rng(8);X=rng.uniform(-.1,.1,(80,3));parts=np.r_[np.zeros(40,int),np.ones(40,int)]
    axis=np.array([0.,0.,1.]);origin=np.zeros(3)
    Y=X.copy();J=joint_transform(axis,origin,np.radians(50));Y[40:]=X[40:]@J[:3,:3].T+J[:3,3]
    Y += [.2,-.1,1]
    pose,q,_,clipped=constrained_fit(X,Y,np.ones(80),parts,axis,origin,'revolute',0,
        np.radians([0,18]),np.eye(3),0.)
    assert 0<=q<=np.radians(18) and clipped
    swung=X.copy();J=joint_transform(axis,origin,q);swung[40:]=X[40:]@J[:3,:3].T+J[:3,3]
    assert np.allclose(pose,rigid_fit(swung,Y,np.ones(80),np.eye(3),0),atol=1e-8)
    # Degenerate line: regularization preserves prior roll without forbidding translation.
    line=np.c_[np.linspace(-.1,.1,40),np.zeros((40,2))]
    p=rigid_fit(line,line+[.01,.02,.03],np.ones(40),np.eye(3),.008)
    assert np.allclose(p[:3,:3],np.eye(3)) and np.allclose(p[:3,3],[.01,.02,.03])
    # Prismatic units and hard bounds remain length-based.
    Y=X.copy();Y[40:]+=axis*.06
    p,q,_,c=constrained_fit(X,Y,np.ones(80),parts,axis,origin,'prismatic',0,[0,.02],np.eye(3),0)
    assert 0<=q<=.02 and c
    owner=SimpleNamespace(joint=dict(joint_type='revolute',part_ids=parts,joint_axis_o=axis,
        joint_origin_o=origin,joint_limits=np.radians([0,18])),coordinate=0.,predicted=0.,
        pinned=False,min_moving_anchors=12,current_frame=0)
    frame=FusionFrame({i:[{}] for i in range(80)},{},{},np.eye(4))
    ctx=FusionContext(1.,18.,15.,6.,2,'linear',1.,'uniform',4.,1.,'mean',.0001,6)
    target=X+[.2,0,0]
    rows=[dict(anchor_id=i,position_o=X[i],position_w=target[i],triangulation_residual_mm=0) for i in range(80)]
    with patch('utils.multiview_geometry.triangulate_anchor_observations',return_value=dict(status='ok',records=rows)):
        bad=StableArticulatedBackend(owner,{}).solve(frame,ctx)
    assert bad.pose is None and bad.fit_status=='stability_rejected'
    assert bad.health['stability_rejection_reason']=='large_body_translation'
    assert not bad.health['theta_observed']
    target=X+[.01,0,0]
    for i,r in enumerate(rows):r['position_w']=target[i]
    with patch('utils.multiview_geometry.triangulate_anchor_observations',return_value=dict(status='ok',records=rows)):
        good=StableArticulatedBackend(owner,{}).solve(frame,ctx)
    assert good.pose is not None and good.health['num_static_inliers']>=12
    assert np.allclose(good.pose[:3,3],[.01,0,0],atol=1e-6)
    assert np.allclose(owner.coordinate,0.)
    # Independent RGB flow says the body is static: do not let a coherent 3D
    # fit walk 10 mm/frame while still calling its 50 mm global step gate a pass.
    owner.flow_prior_health=dict(body_flow_bootstrap=True,body_flow_inliers=100)
    with patch('utils.multiview_geometry.triangulate_anchor_observations',return_value=dict(status='ok',records=rows)):
        guarded=StableArticulatedBackend(owner,dict(flow_consistency_gate=True)).solve(frame,ctx)
    assert guarded.pose is None and guarded.health['flow_body_consistency_gate']
    assert guarded.health['stability_rejection_reason']=='flow_body_translation_disagreement'
    owner.flow_prior_health=dict(body_flow_bootstrap=False,body_flow_inliers=100)
    with patch('utils.multiview_geometry.triangulate_anchor_observations',return_value=dict(status='ok',records=rows)):
        unguarded=StableArticulatedBackend(owner,dict(flow_consistency_gate=True)).solve(frame,ctx)
    assert unguarded.pose is not None and not unguarded.health['flow_body_consistency_gate']
    # The moving arm must not pull the static body's translation/rotation.
    shifted=X.copy();shifted[40:]+=[.025,.01,0]
    body,q,_,_=constrained_fit(X,shifted,np.ones(80),parts,axis,origin,'revolute',0,
        np.radians([-5,18]),np.eye(3),.008,body_from_static=True)
    assert np.allclose(body,np.eye(4),atol=1e-7)
    # A moving-part residual imbalance is a held joint, not an observed fit.
    for i,r in enumerate(rows):
        r['position_w']=X[i]+[.01,0,.012 if parts[i] else 0]
    with patch('utils.multiview_geometry.triangulate_anchor_observations',return_value=dict(status='ok',records=rows)):
        held=StableArticulatedBackend(owner,dict(body_from_static=True)).solve(frame,ctx)
    assert held.pose is not None and held.fit_status=='theta_rejected'
    assert not held.health['theta_observed'] and not held.health['theta_clamped']
    assert held.health['joint_value']==owner.coordinate and held.health['num_moving_inliers']==0
    for i,r in enumerate(rows):r['position_w']=target[i]
    # Upstream surface identities are applied before inlier trimming/counting.
    owner.joint.update(part_surface_samples_o=X,part_surface_offsets=np.array([0,40,80]))
    with patch('utils.multiview_geometry.triangulate_anchor_observations',return_value=dict(status='ok',records=rows)), \
         patch('utils.multiview_geometry.reject_wrong_surface_anchors',return_value=dict(keep_mask=np.zeros(80,bool),num_rejected=80)) as gate:
        rejected=StableArticulatedBackend(owner,{}).solve(frame,ctx)
    assert gate.called and rejected.pose is None
    assert rejected.health['wrong_surface_rejected']==80
    assert rejected.health['stability_rejection_reason']=='insufficient_part_support'
    owner.joint.pop('part_surface_samples_o');owner.joint.pop('part_surface_offsets')
    # A bad third view must not destroy an otherwise consistent stereo anchor.
    K=np.diag([300.,300.,1.]);extr={}
    for cid,t in [('0',[.2,0,0]),('1',[-.2,0,0]),('2',[0,.2,0])]:
        E=np.eye(4);E[:3,3]=t;extr[cid]=E
    prior=np.eye(4);prior[:3,3]=[0,0,1]
    observations={}
    for i,x in enumerate(X):
        observations[i]=[]
        for cid,E in extr.items():
            y=E[:3,:3]@(x+prior[:3,3])+E[:3,3];uv=(K@y)[:2]/y[2]
            if cid=='2':uv+=1000
            observations[i].append(dict(camera_id=cid,uv_curr=uv,position_o=x,valid_flag=True))
    f=FusionFrame(observations,{cid:K for cid in extr},extr,prior)
    good,removed,counts=gate_view_rays(f,owner.joint,0.,.035)
    assert len(good)==80 and all(len(v)==2 for v in good.values())
    assert removed==80 and counts=={'2':80}
    # Exercise the full backend's metre/millimetre boundary, not only the solver.
    # The external joint length, meshes and poses must all scale together.
    answers=[]
    for scale in [1., .001]:
        src=X/scale;dst=src.copy();dst[40:]+=axis*.014/scale
        dst+=np.array([.01,.02,.03])/scale
        own=SimpleNamespace(joint=dict(joint_type='prismatic',part_ids=parts,joint_axis_o=axis,
            joint_origin_o=origin,joint_limits=np.array([0,.02])/scale),coordinate=0.,
            predicted=0.,pinned=False,min_moving_anchors=12,current_frame=0)
        context=FusionContext(scale,18.,15.,6.,2,'linear',1.,'uniform',4.,1.,'mean',.0001,6)
        triangulated=[dict(anchor_id=i,position_o=src[i],position_w=dst[i],
                          triangulation_residual_mm=0) for i in range(80)]
        prism_backend=StableArticulatedBackend(own,dict(part_support_gate=True))
        with patch('utils.multiview_geometry.triangulate_anchor_observations',
                   return_value=dict(status='ok',records=triangulated)):
            answer=prism_backend.solve(frame,context)
        assert answer.pose is not None and answer.health['theta_observed']
        assert 0<=answer.health['joint_value']*scale<=.02
        answers.append((answer.pose[:3,3]*scale,answer.health['joint_value']*scale))
        prism_backend.commit_support('ok',answer.health)
        own.current_frame=1
        sparse=triangulated[:14]+triangulated[40:]
        with patch('utils.multiview_geometry.triangulate_anchor_observations',
                   return_value=dict(status='ok',records=sparse)):
            lost=prism_backend.solve(frame,context)
        assert lost.pose is None and lost.health['stability_rejection_reason']=='static_support_drop'
    assert np.allclose(answers[0][0],answers[1][0],atol=1e-6)
    assert np.isclose(answers[0][1],answers[1][1],atol=1e-6)
    assert np.isclose(answers[0][1],.014,atol=1e-5)
    # Count collapse is tested causally against previous accepted evidence.
    # An otherwise exact fit with >12 static anchors can still lose most of its
    # body support. Joint-only rejection must preserve an actually measured body.
    own=SimpleNamespace(joint=dict(joint_type='revolute',part_ids=parts,joint_axis_o=axis,
        joint_origin_o=origin,joint_limits=np.radians([-5,18])),coordinate=0.,predicted=0.,
        pinned=False,min_moving_anchors=12,current_frame=120)
    backend=StableArticulatedBackend(own,dict(part_support_gate=True,body_from_static=True))
    def supported(ids,angle=0):
        dst=X.copy();j=joint_transform(axis,origin,angle)
        dst[40:]=X[40:]@j[:3,:3].T+j[:3,3];dst+=[.01,0,0]
        return [dict(anchor_id=i,position_o=X[i],position_w=dst[i],
            triangulation_residual_mm=0,num_support_views=4,mean_ray_angle_deg=25.) for i in ids]
    def solve_rows(values):
        with patch('utils.multiview_geometry.triangulate_anchor_observations',
                   return_value=dict(status='ok',records=values)):
            answer=backend.solve(frame,ctx)
        backend.commit_support('ok' if answer.pose is not None else 'fallback_prior',answer.health)
        return answer
    first=solve_rows(supported(range(80)))
    assert first.pose is not None and first.health['body_observed']
    assert backend.support_reference('static')==40 and backend.support_reference('moving')==40
    assert first.health['static_inlier_support_views_median']==4
    assert first.health['moving_inlier_parallax_deg_median']==25
    solve_rows(supported(range(80)))
    assert len(backend.support_history['static'])==1,'Repeated same-frame fits must not duplicate history'
    own.current_frame=119  # Reverse replay is supported; no increasing-index assumption.
    before=list(backend.support_history['static'])
    collapsed=solve_rows(supported(list(range(14))+list(range(40,80))))
    assert collapsed.pose is None and collapsed.health['stability_rejection_reason']=='static_support_drop'
    assert collapsed.health['num_static_inliers']==14 and not collapsed.health['body_observed']
    assert list(backend.support_history['static'])==before,'Rejected support must not poison reference'
    disabled=StableArticulatedBackend(own,dict(body_from_static=True))
    disabled.support_history['static'].append(40)
    with patch('utils.multiview_geometry.triangulate_anchor_observations',
               return_value=dict(status='ok',records=supported(list(range(14))+list(range(40,80))))):
        unchanged=disabled.solve(frame,ctx)
    assert unchanged.pose is not None,'The disabled option must retain the absolute-count path'
    own.current_frame=118
    moving_weak=solve_rows(supported(list(range(40))+list(range(40,54)),np.radians(8)))
    assert moving_weak.pose is not None and moving_weak.fit_status=='theta_rejected'
    assert moving_weak.health['joint_rejection_reason']=='moving_support_drop'
    assert moving_weak.health['body_observed'] and not moving_weak.health['theta_observed']
    assert moving_weak.health['joint_value']==0 and moving_weak.health['num_moving_inliers']==0
    assert np.allclose(moving_weak.pose[:3,3],[.01,0,0],atol=1e-7)
    assert backend.support_reference('moving')==40
    own.current_frame=117
    restored=solve_rows(supported(range(80),np.radians(8)))
    assert restored.pose is not None and restored.health['theta_observed']
    assert np.isclose(restored.health['joint_value'],np.radians(8),atol=1e-5)
    backend.reset_support_history()
    assert backend.support_reference('static') is None and backend.support_reference('moving') is None
    assert backend.support_last_frame is None
    backend.commit_support('fallback_prior',restored.health)
    assert backend.support_reference('static') is None,'Upstream policy rejection must not teach history'
    assert not restored.health['body_observed'],'A rejected candidate is not a body observation'
    for config in [dict(min_static_support_ratio=1.1),dict(support_reference_window=1.5),
                   dict(support_reference_update_ratio=.2),dict(part_support_gate='yes')]:
        try:StableArticulatedBackend(own,config)
        except ValueError:pass
        else:raise AssertionError(f'Invalid config accepted: {config}')
    print('PASS: bounded joint/body consistency, rotation prior, metre prismatic bounds, real rejection, valid fit')

if __name__=='__main__':main()
