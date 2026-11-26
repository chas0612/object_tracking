"""CPU-only synthetic multiview real-image motion-prior contract test."""
from pathlib import Path
from types import SimpleNamespace
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'autodex/perception/thirdparty/MV-GoTrack-next')]
from autodex.tracking.body_flow_bootstrap import BodyFlowBootstrap

def main():
    cv2.setNumThreads(2)
    rng=np.random.default_rng(4);X=rng.uniform([-.12,-.1,-.02],[.12,.1,.02],(100,3)).astype('float32')
    K=np.tile(np.array([[500.,0,128],[0,500.,128],[0,0,1]]),(3,1,1))
    E=np.tile(np.eye(4),(3,1,1));E[0,0,3]=-.1;E[1,0,3]=.1;E[2,1,3]=-.1
    owner=SimpleNamespace(_anchor_cache={1:dict(positions_o=X,part_ids=np.zeros(100,int))})
    T=np.eye(4);T[2,3]=1
    def images(pose):
        world=X@pose[:3,:3].T+pose[:3,3];out=[]
        for k,e in zip(K,E):
            y=world@e[:3,:3].T+e[:3,3];uv=y@k.T;uv=uv[:,:2]/uv[:,2:3]
            image=np.zeros((256,256,3),np.uint8)
            for i,(x,y) in enumerate(uv):cv2.circle(image,(round(x),round(y)),2,(80+i%170,)*3,-1)
            out.append(image)
        return out
    flow=BodyFlowBootstrap(owner,K,E);assert flow.propose(images(T))[0] is None
    flow.update(images(T),T)
    moved=T.copy();moved[:3,3]+=[.005,.003,0]
    estimated,health=flow.propose(images(moved))
    assert estimated is not None and health['body_flow_bootstrap']
    assert np.linalg.norm(estimated[:3,3]-moved[:3,3])<.005,(estimated,health)
    assert np.linalg.det(estimated[:3,:3])>.999
    assert np.allclose(flow.pose,T),'propose must not mutate previous state'
    owner.coordinate=0.
    owner.joint=dict(part_surface_samples_o=X,part_surface_offsets=np.array([0,len(X),len(X)]),
        joint_axis_o=np.array([0.,0.,1.]),joint_origin_o=np.zeros(3),joint_type='revolute')
    visible=BodyFlowBootstrap(owner,K,E,visible_only=True,min_views=3)
    visible.update(images(T),T)
    candidate,h=visible.propose(images(moved))
    assert candidate is not None and h['body_flow_bootstrap'],h
    assert np.linalg.norm(candidate[:3,3]-moved[:3,3])<.005
    print('PASS: real-image multiview flow measures translation; valid SE3; explicit proposal/update state')

if __name__=='__main__':main()
