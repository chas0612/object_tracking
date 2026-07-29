# MV-GoTrack articulated-object patch

Tracks an object made of two rigid parts joined by one revolute joint: six degrees
of freedom for the body and one for the joint, recovered together, per frame.

Apply **on top of** `MV-GoTrack-offline-capture.patch`, from the root of the
approved private MV-GoTrack checkout at commit
`a9f033734c0bdf2d191265d22ea732a914c861f6`.

```bash
cd /path/to/MV-GoTrack
git apply --check /path/to/autodex/patches/MV-GoTrack-offline-capture.patch
git apply       /path/to/autodex/patches/MV-GoTrack-offline-capture.patch
git apply --check /path/to/autodex/patches/MV-GoTrack-articulated.patch
git apply       /path/to/autodex/patches/MV-GoTrack-articulated.patch

# No GPU, no capture, no checkpoint needed.
python tests/test_articulated_fit.py
python tests/test_articulated_anchor_bank.py       # needs the blue_plastic_box parts
python tests/test_articulated_runtime_wiring.py    # needs the blue_plastic_box parts
```

Every change is inert for a rigid object. An anchor bank without a joint takes the
original code path everywhere, and a bank saved before this patch loads as one
static part.

## What changed, and why there

The anchor tracker is *triangulate first, fit later*, which confines articulation
to two places. Once the anchors are swung to the current angle the object is rigid
again, so flow lookup, cross-view grouping and triangulation cannot tell a moved
anchor from one that was always there — they are untouched.

| File | Change |
|---|---|
| `utils/multiview_geometry.py` | `joint_transform`, `fit_joint_angle_weighted`, `fit_articulated_transform_weighted`, `compute_articulated_fit_residuals`, `robust_fit_articulated_pose_from_anchors` |
| `utils/anchor_bank.py` | `sample_articulated_mesh_anchors`, `fuse_articulated_parts`, `articulation_from_anchor_bank`; `part_ids` and the joint carried through save/load and the GoTrack unit conversion |
| `utils/anchor_tracking.py` | `pose_anchors_at_joint_angle`; `project_anchors_to_view` takes the joint and the current angle |
| `utils/renderer_nvdiffrast.py` | `set_object_joint`, `set_object_joint_angle`; `_get_object_buffers` poses the moving vertices, cached per angle |
| `archive/run_multiview_gotrack_anchor_online.py` | angle reaches the projection; `theta_*` fields on the world record |
| `archive/…_multi_object.py` | `--articulation-json`, `--init-theta-deg`, `--anchor-fit-min-moving-anchors`; joint registered on the renderer; angle carried between frames |
| `tests/` | three self-contained test scripts |

## The fit

```
argmin_{T, theta}  sum_body w ||T X - Y||^2  +  sum_lid w ||T J(theta) X - Y||^2
```

Neither variable has a closed form alone; each does given the other. Fix the angle
and the moving anchors pre-swing into an ordinary weighted Kabsch. Fix the pose and
the moving targets pull back into the body frame, where one degree of freedom about
a known axis is an arctangent. Alternating exact solves is monotone, and Aitken
extrapolation cuts the linear tail — on one geometry it still had 0.009 deg left
after forty rounds without it.

No general-purpose optimiser, deliberately. The silhouette stage of this work lost
a long time to a Powell search that stopped early on a bumpy surface and returned
answers 8.9 deg apart from equivalent starting points.

Measured cost on a 12-camera rig: **0.012 s/frame for triangulation and fit
together, 3.3 % of a 0.36 s frame.**

## Things that fail silently if you get them wrong

- **Units.** The joint origin is a point and scales into GoTrack units; the axis is
  a direction and must not. Neither mistake raises — the hinge simply moves a
  thousand times too far away. Both copies are stored explicitly and covered by a
  test.
- **Vertex order.** Anchor `vertex_indices` are offset onto the parts concatenated
  in bank order, so the renderer's mesh must be that same concatenation.
  `fuse_articulated_parts` builds it rather than trusting a pre-fused asset, whose
  parts could be in the other order or welded.
- **Double-swinging.** `project_anchors_to_view` returns `anchors_o` canonical and
  poses only a copy, because the fit expects unposed anchors and the angle
  separately. A test asserts the canonical array is unchanged.
- **Reporting window.** An arctangent returns (-pi, pi]. This hinge opens past 180
  deg, so 215 comes back as -145 and a lower limit of zero clamps it to nothing.
  `theta_reference_rad` names the half-turn to report in; pass the previous frame's
  angle when tracking.
- **Seeding the angle.** A lid seeded closed renders closed templates, the depth
  visibility test then rejects every anchor on the open lid, and the angle can never
  recover while the body keeps tracking perfectly. The angle therefore travels in
  the init pose records next to the pose.

## Two states that are not measurements

Reported rather than smoothed over, because a reader cannot tell either from a real
value by looking at the angle:

- `theta_observed = false` — too few anchors on the moving part this frame. The
  previous angle is held, not re-measured. The runtime only advances its stored
  angle on an observed frame; writing a held value back would make the next frame
  unable to tell the difference, and a run of occluded frames would read as a
  confidently static lid.
- `theta_clamped = true` — a joint limit bound and the estimate is censored.

That second one is not hypothetical here. `joint.json` for `blue_plastic_box`
reports a range ending at 205.8 deg, which is how far the *open scan* was opened,
not how far the hinge goes. The lid reaches 213-215 deg, confirmed on two separate
captures by two independent methods. Clipped at the scanned value, 77 of 203 frames
pinned to the bound exactly and looked like measurements.
