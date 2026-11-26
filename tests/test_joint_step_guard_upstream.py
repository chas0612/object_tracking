"""CPU integration against the pinned upstream articulated solver and policy."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'autodex/perception/thirdparty/MV-GoTrack-next'))
from autodex.tracking.joint_step_guard import JointStepGuard
from evaluation.dexycb_tracking import articulated
from multiview_solver.fusion_backends import FusionFrame, FusionContext
from utils.anchor_tracking import pose_anchors_at_joint_angle


def main():
    source = np.random.default_rng(0).uniform(-.1, .1, (48, 3))
    parts = np.array([0] * 32 + [1] * 16)
    axis, origin = np.array([0., 0., 1.]), np.zeros(3)
    body = np.eye(4); body[:3, 3] = [.2, -.1, 1.]
    target = pose_anchors_at_joint_angle(source, parts, axis, origin, np.radians(65)) + body[:3, 3]
    owner = SimpleNamespace(joint=dict(joint_type='revolute', part_ids=parts,
                                     joint_axis_o=axis, joint_origin_o=origin,
                                     theta_limits_rad=np.radians([-90, 120]),
                                     joint_limits=np.radians([-90, 120])),
                            coordinate=np.radians(20), predicted=np.radians(20),
                            pinned=False, wrong_surface_margin=0,
                            min_moving_anchors=12, residual_ratio=3)
    frame = FusionFrame({i: [{}] for i in range(48)}, {}, {}, body)
    ctx = FusionContext(1., 15., 15., 15., 2, 'linear', 1., 'uniform', 2., 1., 'median', 0., 6)

    def triangulate(**kwargs):
        return dict(status='ok', records=[dict(anchor_id=i, position_o=source[i],
                                              position_w=target[i], triangulation_residual_mm=0.)
                                         for i in kwargs['observations_by_anchor']])

    backend = articulated.ArticulatedBackend(owner)
    with patch.object(articulated, 'triangulate_anchor_observations', triangulate):
        raw = backend.solve(frame, ctx)
        assert raw.health['theta_observed'] and np.isclose(raw.health['joint_value'], np.radians(65), atol=1e-5)
        held = JointStepGuard(backend, owner, np.radians(10)).solve(frame, ctx)
    assert np.allclose(held.pose, body, atol=1e-6)
    assert held.fit_status == 'theta_rejected'
    assert held.health['num_triangulated'] == 32
    assert not held.health['theta_observed'] and held.health['joint_value'] == owner.coordinate
    assert articulated.ArticulatedPolicy.from_config({}).decide('triangulate', held) == (True, 'accepted_body_joint_held')
    assert not owner.pinned and owner.predicted == owner.coordinate
    print('PASS: upstream articulated fit -> reject 45-degree update -> static body refit -> accepted held joint')


if __name__ == '__main__':
    main()
