"""CPU regressions for a gate that rejects joint updates, never clips them."""
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodex.tracking.joint_step_guard import JointStepGuard


@dataclass
class Frame:
    observations_by_anchor: dict = field(default_factory=lambda: {0: ['body'], 1: ['moving']})


@dataclass
class Result:
    pose: object = field(default_factory=lambda: np.eye(4))
    fit_status: str = 'ok'
    health: dict = field(default_factory=dict)


def main():
    owner = SimpleNamespace(joint={'joint_type': 'revolute', 'part_ids': np.array([0, 1])},
                            coordinate=np.radians(20), predicted=np.radians(22), pinned=False)

    class Backend:
        name = 'triangulate'
        measured = 21
        fail = False
        raise_error = False
        observed = True
        calls = 0
        def solve(self, frame, ctx):
            self.calls += 1
            if not owner.pinned:
                return Result(health={'joint_value': np.radians(self.measured),
                                      'theta_observed': self.observed, 'theta_clamped': self.measured == -25})
            assert frame.observations_by_anchor == {0: ['body']}
            assert owner.predicted == owner.coordinate
            if self.raise_error:
                raise RuntimeError('static solver failed')
            return Result(pose=None if self.fail else np.eye(4),
                          fit_status='too_few_anchors' if self.fail else 'ok',
                          health={'num_inlier_anchors': 3})

    backend = Backend()
    guard = JointStepGuard(backend, owner, np.radians(10))
    result = guard.solve(Frame(), None)
    assert backend.calls == 1 and result.health['theta_observed']
    backend.measured = -25
    result = guard.solve(Frame(), None)
    assert backend.calls == 3 and result.fit_status == 'theta_rejected'
    assert result.health['joint_value'] == owner.coordinate
    assert not result.health['theta_observed'] and not result.health['theta_clamped']
    assert result.health['rejected_candidate_theta_clamped']
    assert result.health['joint_source'] == 'step_guard'
    assert np.isclose(result.health['rejected_joint_delta_deg'], -45)
    assert not owner.pinned and np.isclose(owner.predicted, np.radians(22))
    backend.fail = True
    result = guard.solve(Frame(), None)
    assert result.pose is None and result.fit_status == 'too_few_anchors'
    backend.raise_error = True
    try:
        guard.solve(Frame(), None)
        raise AssertionError('Expected failure')
    except RuntimeError:
        assert not owner.pinned and np.isclose(owner.predicted, np.radians(22))
    backend.raise_error = backend.fail = False
    backend.observed = False
    before = backend.calls
    assert not guard.solve(Frame(), None).health['theta_observed']
    assert backend.calls == before + 1
    # Explicit anchors bypass the gate even for large changes.
    owner.pinned = True
    backend.solve = lambda frame, ctx: Result(health={'joint_value': np.radians(90), 'theta_observed': True})
    assert guard.solve(Frame(), None).health['theta_observed']
    for cap in (0, -1, float('nan')):
        try:
            JointStepGuard(backend, owner, cap)
            raise AssertionError('Expected invalid cap')
        except ValueError:
            pass
    owner.joint['joint_type'] = 'prismatic'
    try:
        JointStepGuard(backend, owner, 1)
        raise AssertionError('Expected wrong joint type')
    except ValueError:
        pass
    print('PASS: pass-through, static-only rejection, failure preservation, restoration, anchor bypass')


if __name__ == '__main__':
    main()
