# Articulated case-study notes

## 2026-08-21 — 0820 capture

### Red bowl, episode 3

- Result: `articulated_runs/0820_body_sparse_angular_v1`
- The targeted sparse angular correction covers frames 284–316.
- Manual review found that the articulated part begins to drift slightly again
  after frame 316, while the rigid body pose remains good.
- The current result is intentionally left unchanged. If this sequence is revisited,
  extend the sparse angular anchor window past frame 316 instead of modifying the
  body trajectory.

### Blue plastic box, episode 0

- Result: `articulated_runs/0820_body_sparse_angular_v1`
- Manual review judged the body-only plus sparse angular correction broadly good.
- Frame 92's isolated zero-degree stereo alias was rejected before trajectory
  interpolation.
