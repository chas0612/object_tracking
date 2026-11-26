"""Opt-in real-RGB motion prior: static-anchor LK + multiview triangulation.

This is a local, fallible measurement, never an absolute pose or ground truth.
"""
import cv2
import numpy as np
from autodex.tracking.articulated_stability import rigid_fit

class BodyFlowBootstrap:
    def __init__(self, owner, K, E, radius_m=.008,visible_only=False,min_views=2):
        self.K=np.asarray(K);self.E=np.asarray(E);self.radius=radius_m
        bank=owner._anchor_cache[1];ids=np.flatnonzero(bank['part_ids']==0)
        self.points=np.asarray(bank['positions_o'])[ids]
        self.owner=owner;self.bank=bank;self.visible_only=visible_only;self.min_views=min_views
        self.previous=None;self.pose=None
    def update(self, images, pose):
        self.previous=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in images]
        self.pose=np.asarray(pose,float).copy()
    def propose(self, images):
        if self.previous is None:return None,dict(body_flow_bootstrap=False)
        from utils.multiview_geometry import triangulate_anchor_observations
        table={};prior=self.pose;world=self.points@prior[:3,:3].T+prior[:3,3]
        surface=None
        if self.visible_only:
            from utils.multiview_geometry import joint_transform
            joint=self.owner.joint;surface=np.asarray(joint['part_surface_samples_o']).copy()
            offset=int(joint['part_surface_offsets'][1])
            J=joint_transform(joint['joint_axis_o'],joint['joint_origin_o'],self.owner.coordinate,
                              joint_type=joint['joint_type'])
            surface[offset:]=surface[offset:]@J[:3,:3].T+J[:3,3]
            surface=surface@prior[:3,:3].T+prior[:3,3]
        for v,(rgb,old,K,E) in enumerate(zip(images,self.previous,self.K,self.E)):
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            cam=world@E[:3,:3].T+E[:3,3];uv=cam@K.T;uv=uv[:,:2]/uv[:,2:3]
            h,w=old.shape
            inside=(cam[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>15)&(uv[:,0]<w-15)&(uv[:,1]>15)&(uv[:,1]<h-15)
            if surface is not None:
                # Cheap mesh self-occlusion test. This does NOT detect an external
                # hand; it only prevents LK from attaching hidden/back anchors to
                # whatever happens to be in front of the projected surface.
                sc=surface@E[:3,:3].T+E[:3,3];sp=sc@K.T;sp=sp[:,:2]/sp[:,2:3]
                good=(sc[:,2]>0)&np.isfinite(sp).all(1)
                pixel=np.rint(sp[good]).astype(int);z=sc[good,2]
                good=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
                depth=np.full(h*w,np.inf,dtype=np.float32)
                np.minimum.at(depth,pixel[good,1]*w+pixel[good,0],z[good])
                depth=cv2.erode(depth.reshape(h,w),np.ones((3,3),np.uint8))
                indices=np.flatnonzero(inside);pixel=np.rint(uv[indices]).astype(int)
                visible=cam[indices,2]<=depth[pixel[:,1],pixel[:,0]]+.004
                inside[indices]&=visible
            indices=np.flatnonzero(inside)
            if len(indices)<6:continue
            p=uv[indices].astype('float32').reshape(-1,1,2)
            q,ok,err=cv2.calcOpticalFlowPyrLK(old,gray,p,None,winSize=(25,25),maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
            if q is None:continue
            back,okback,_=cv2.calcOpticalFlowPyrLK(gray,old,q,None,winSize=(25,25),maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
            if back is None:continue
            valid=ok.ravel().astype(bool)&okback.ravel().astype(bool)
            valid&=np.linalg.norm(back.reshape(-1,2)-p.reshape(-1,2),axis=1)<1.5
            valid&=err.ravel()<25
            for index,point,error in zip(indices[valid],q.reshape(-1,2)[valid],err.ravel()[valid]):
                if not np.isfinite(point).all():continue
                table.setdefault(int(index),[]).append(dict(camera_id=str(v),position_o=self.points[index],
                    uv_curr=point,confidence=float(np.exp(-error/10)),valid_flag=True))
        tri=triangulate_anchor_observations(table,{str(i):k for i,k in enumerate(self.K)},
            {str(i):e for i,e in enumerate(self.E)},min_views=self.min_views,external_unit_scale_to_meter=1.)
        rows=[r for r in tri.get('records',[]) if r['triangulation_residual_mm']<12]
        if len(rows)<12:return None,dict(body_flow_bootstrap=False,body_flow_anchors=len(rows))
        X=np.array([r['position_o'] for r in rows]);Y=np.array([r['position_w'] for r in rows])
        W=np.array([r['mean_confidence'] for r in rows]);mask=np.ones(len(rows),bool)
        for _ in range(3):
            if mask.sum()<12:return None,dict(body_flow_bootstrap=False,body_flow_anchors=len(rows))
            T=rigid_fit(X[mask],Y[mask],W[mask],prior[:3,:3],self.radius)
            residual=np.linalg.norm(X@T[:3,:3].T+T[:3,3]-Y,axis=1);mask=residual<.018
        angle=np.degrees(np.arccos(np.clip((np.trace(prior[:3,:3].T@T[:3,:3])-1)/2,-1,1)))
        distance=np.linalg.norm(prior[:3,3]-T[:3,3]);good=mask.sum()>=12 and angle<40 and distance<.06
        return (T if good else None),dict(body_flow_bootstrap=bool(good),body_flow_anchors=len(rows),
            body_flow_inliers=int(mask.sum()),body_flow_rotation_deg=float(angle),body_flow_translation_m=float(distance))
