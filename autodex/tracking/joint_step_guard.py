"""Opt-in joint outlier rejection around the pinned articulated backend.

This is a measurement gate, not an angle clip: an excessive one-frame update
is discarded, and the body is re-fit using only the static part. The last
observed coordinate is retained and explicitly reported as unobserved.
"""
from dataclasses import replace

import numpy as np


class JointStepGuard:
    def __init__(self, backend, owner, max_step_rad):
        if not np.isfinite(max_step_rad) or max_step_rad <= 0:
            raise ValueError("Joint step cap must be finite and positive")
        if owner.joint["joint_type"] != "revolute":
            raise ValueError("Angular joint step guard requires a revolute joint")
        self.backend, self.owner, self.max_step_rad = backend, owner, max_step_rad
        self.name = backend.name

    def solve(self, frame, ctx):
        candidate = self.backend.solve(frame, ctx)
        owner = self.owner
        health = candidate.health
        # Explicit external anchors remain authoritative. Unobserved candidates
        # are already held by upstream, and must not be relabelled as new fits.
        if owner.pinned or candidate.pose is None or not health.get("theta_observed", False):
            return candidate
        measured = float(health["joint_value"])
        previous = float(owner.coordinate)
        delta = measured - previous
        if np.isfinite(delta) and abs(delta) <= self.max_step_rad:
            return candidate
        parts = owner.joint["part_ids"]
        static = {aid: obs for aid, obs in frame.observations_by_anchor.items()
                  if 0 <= int(aid) < len(parts) and parts[int(aid)] == 0}
        saved = owner.pinned, owner.predicted
        try:
            # Reuse the upstream rigid fit with the coordinate held; do not let
            # the rejected moving-part correspondences pull the body with them.
            owner.pinned, owner.predicted = True, previous
            held = self.backend.solve(replace(frame, observations_by_anchor=static), ctx)
        finally:
            owner.pinned, owner.predicted = saved
        held.health.update(joint_value=previous, theta_rad=previous,
                           theta_deg=float(np.degrees(previous)), theta_observed=False,
                           theta_rejected=True, theta_clamped=False, joint_source="step_guard",
                           joint_step_rejected=True,
                           rejected_joint_value=measured,
                           rejected_joint_delta_deg=float(np.degrees(delta)),
                           rejected_candidate_theta_clamped=bool(health.get("theta_clamped", False)),
                           max_joint_step_deg=float(np.degrees(self.max_step_rad)),
                           rejected_candidate_fit_status=candidate.fit_status)
        # If the static fit failed, retain its real failure status / None pose.
        # Never turn insufficient static support into an accepted held pose.
        if held.pose is not None and held.fit_status in {"ok", "no_inlier_refit"}:
            held.fit_status = "theta_rejected"
        return held
