"""CPU tests: deterministic consensus, ambiguity, and external-unit boundaries."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'autodex/perception/thirdparty/MV-GoTrack-next')]
from autodex.tracking.robust_rays import consensus_point,triangulate_consensus
from autodex.tracking.articulated_stability import StableArticulatedBackend
from multiview_solver.fusion_backends import FusionFrame,FusionContext


def main():
    centres=np.array([[-.4,0,0],[.4,0,0],[0,.4,0],[0,-.4,0],[-.3,.3,0],[.3,.3,0]])
    target=np.array([.02,.01,1.])
    clean=target-centres
    a=consensus_point(centres,clean,np.ones(6),threshold=.008)
    assert a is not None and a['mask'].all() and np.allclose(a['point'],target,atol=1e-10)
    poisoned=clean.copy();poisoned[4:]=target+[.06,0,0]-centres[4:]
    a=consensus_point(centres,poisoned,np.ones(6),threshold=.008)
    assert a is not None and np.array_equal(a['mask'],[1,1,1,1,0,0])
    assert np.allclose(a['point'],target,atol=1e-10)
    # Competing equal-mass 3-view explanations are unobservable, not a winner.
    ambiguous=clean.copy();ambiguous[3:]=target+[.1,0,0]-centres[3:]
    assert consensus_point(centres,ambiguous,np.ones(6),threshold=.002) is None
    assert consensus_point(centres,np.tile([0,0,1.],(6,1)),np.ones(6),threshold=.008) is None
    assert consensus_point(centres[:2],clean[:2],np.ones(2),threshold=.008) is None
    assert consensus_point(centres,clean,np.zeros(6),threshold=.008) is None
    bad=clean.copy();bad[0,0]=np.nan
    assert consensus_point(centres,bad,np.ones(6),threshold=.008) is None
    for scale in [1.,.001]:
        observations=[];Ks={};Es={}
        K=np.diag([500.,500.,1.])
        for i,c in enumerate(centres):
            cid=str(i);E=np.eye(4);E[:3,3]=-c/scale;Ks[cid]=K;Es[cid]=E
            y=target-c
            if i>=4:y=target+[.06,0,0]-c
            uv=(K@y)[:2]/y[2]
            observations.append(dict(camera_id=cid,uv_curr=uv,confidence=1.,position_o=np.zeros(3)))
        t=triangulate_consensus({7:observations},Ks,Es,min_views=3,external_unit_scale_to_meter=scale)
        assert len(t['records'])==1
        r=t['records'][0]
        assert np.allclose(r['position_w']*scale,target,atol=1e-6)
        assert r['support_camera_ids']==['0','1','2','3'] and r['num_support_views']==4
        assert r['triangulation_residual_mm']<1e-6
        assert t['robust_ray_health']['consensus_rejected_by_view']=={'4':1,'5':1}
    rng=np.random.default_rng(30)
    src=rng.uniform([-.08,-.08,.9],[.08,.08,1.1],(80,3));parts=np.r_[np.zeros(40,int),np.ones(40,int)]
    answers=[]
    for scale in [1.,.001]:
        points=src/scale;target=points+np.array([.001,.002,.008])/scale
        target[40:,2]+=.014/scale;obs={};Ks={};Es={};K=np.diag([500.,500.,1.])
        for i,c in enumerate(centres):
            cid=str(i);E=np.eye(4);E[:3,3]=-c/scale;Ks[cid]=K;Es[cid]=E
            for aid,x in enumerate(target):
                y=x-c/scale
                if i==5:y=y+np.array([.03,0,0])/scale
                uv=(K@y)[:2]/y[2]
                obs.setdefault(aid,[]).append(dict(camera_id=cid,uv_curr=uv,position_o=points[aid],confidence=1.))
        owner=SimpleNamespace(joint=dict(joint_type='prismatic',part_ids=parts,joint_axis_o=np.array([0.,0.,1.]),
            joint_origin_o=np.zeros(3),joint_limits=np.array([0.,.02])/scale),coordinate=0.,predicted=0.,
            pinned=False,min_moving_anchors=3,current_frame=10)
        frame=FusionFrame(obs,Ks,Es,np.eye(4))
        ctx=FusionContext(scale,18.,15.,6.,2,'linear',1.,'uniform',4.,1.,'mean',.0001,6)
        profile=dict(robust_ray_consensus=True,body_from_static=True,min_view_support=3)
        fit=StableArticulatedBackend(owner,profile).solve(frame,ctx)
        assert fit.pose is not None and fit.health['theta_observed']
        assert np.isclose(fit.health['joint_value']*scale,.014,atol=1e-5)
        assert (fit.health['consensus_rejected_by_view'].get('5',0)
                +fit.health['ray_prior_rejected_by_view'].get('5',0))==80
        answers.append(fit.pose[:3,3]*scale)
    assert np.allclose(answers[0],answers[1],atol=1e-6)
    print('robust ray consensus tests passed')


if __name__=='__main__':main()
