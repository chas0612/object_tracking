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
static part without the surface samples the wrong-surface test needs, which disables
that test rather than silently weakening it.

That claim was false once: an edit meant for the articulated fit deleted a line from
the *rigid* one, and all three articulated test files passed while stock rigid GoTrack
could not run at all. It surfaced only when an experiment happened to exercise the
rigid path. There is still no test covering it.

## What changed, and why there

The anchor tracker is *triangulate first, fit later*, which confines articulation
to two places. Once the anchors are swung to the current angle the object is rigid
again, so flow lookup, cross-view grouping and triangulation cannot tell a moved
anchor from one that was always there — they are untouched.

| File | Change |
|---|---|
| `utils/multiview_geometry.py` | `joint_transform`, `fit_joint_angle_weighted`, `fit_articulated_transform_weighted`, `compute_articulated_fit_residuals`, `robust_fit_articulated_pose_from_anchors`, `reject_wrong_surface_anchors` |
| `utils/anchor_bank.py` | `sample_articulated_mesh_anchors`, `fuse_articulated_parts`, `articulation_from_anchor_bank`; `part_ids`, per-part surface samples and the joint carried through save/load and the GoTrack unit conversion |
| `utils/anchor_tracking.py` | `pose_anchors_at_joint_angle`; `project_anchors_to_view` takes the joint and the current angle |
| `utils/renderer_nvdiffrast.py` | `set_object_joint`, `set_object_joint_angle`; `_get_object_buffers` poses the moving vertices, cached per angle |
| `archive/run_multiview_gotrack_anchor_online.py` | angle reaches the projection; `theta_*` fields on the world record |
| `archive/…_multi_object.py` | `--articulation-json`, `--init-theta-deg`, `--anchor-fit-min-moving-anchors`, `--theta-extrapolate-max-deg`, `--anchor-wrong-surface-margin`, `--dump-anchor-target-frames`; joint registered on the renderer; angle predicted, carried and measured between frames |
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

## The angle running away, and what it turned out to be

On one of seven held-out captures the angle walked from 215 deg to the 260 deg
ceiling while the lid was actually closing, and stayed pinned there for the
remaining 75 frames. Anchor counts stayed healthy throughout (101-140 on the moving
part), so `theta_observed` never fired. The body pose was dragged 7 deg along with
it through the shared Kabsch, smoothly, so trajectory smoothness did not reveal it
either. Two things made this diagnosable, and both are worth keeping.

**An independent reference.** Tracking the two parts as ordinary rigid objects — same
frames, same cameras, same seed, no joint — recovered the true trajectory on the first
attempt: 231/231 frames each, mutually consistent as a hinge to 0.37 deg, and the
angle read off them matched the footage. That reference is what turned "the tracker
looks wrong" into a measurement. It is not the production design (a rigid pair has
five degrees of freedom the object does not have) but it is the ground truth to check
against, and `src/process/articulated/which_surface.py` uses it.

**Asking where the anchors went.** Against that reference, the moving part's anchors
were 1.8 mm off their own surface while healthy, and by the time the angle had
settled, 100 % of them were nearer the *static* part's surface, sitting 2.5-6.3 mm
off it. They had migrated onto the body and were tracking it faithfully. That also
explains the ceiling: folding the lid back toward the body is what large angles do
(the modelled lid sits 42.8 mm from the body at 215 deg, 13.6 mm at 260), so 260 was
the fit correctly chasing anchors that were on the wrong part. Raising the ceiling
would not have stopped it, it would have let it continue toward 360.

### The cause was prediction lag, not a bad fit

The lid closes at about 7 deg/frame. Anchor projection and template rendering both
used the *previous* frame's angle, so every frame started a full frame of motion
behind the object and the flow searched from where the lid had been.
`--theta-extrapolate-max-deg` (project and render at the angle extrapolated from the
last two measurements, capped) removes that:

| capture 6 | angle error vs. the rigid pair | body rotation error |
|---|---|---|
| baseline | median 6.2 deg, max 260 | 7.27 deg |
| wrong-surface rejection only | median 6.0 deg, max 260 | 5.15 deg |
| **extrapolation** | **median 0.4 deg, max 20.3** | **0.29 deg** |

The cap matters more than the gain: predicting short leaves the small lag that was
already there, while predicting long puts the template where the object never was.

### Held-out result

All seven captures the method was not developed on, 22 cameras, one FoundationPose
silhouette seed at frame 40, `--theta-extrapolate-max-deg 15`:

| capture | frames | angle | expected motion |
|---|---|---|---|
| 0 | 80/80 | 213.6 → 214.8 | open, still |
| 1 | 249/249 | 213.3 → 2.9 | open → closed |
| 3 | 214/214 | 213.2 → 2.2 | open → closed |
| 4 | 183/183 | 1.3 → 214.7 | closed → open |
| 5 | 224/224 | 213.6 → 2.7 | open → closed |
| 6 | 231/231 | 1.5 → 215.3 → 2.1 | closed → open → closed |
| 7 | 289/289 | 0.9 → 208.9 → 2.2 | closed → open → closed |

Every frame fitted, no clamping except 2 and 5 frames on the two captures that reach
furthest open. The closed end reads 0.6-2.9 deg rather than 0 because the lid rests
where it sits without being pressed.

Capture 0 needs camera `22645029` excluded: its recording there is one frame long, and
the frame count is the minimum across cameras, so including it silently truncates the
run to a single frame. Worth checking per capture rather than assuming, since the
symptom is a completed run with almost no output.

### Running it

```bash
--articulation-json joint.json --theta-extrapolate-max-deg 15
# defaults, no need to pass: --anchor-wrong-surface-margin 0.02 --theta-residual-ratio 3
```

The angle also has to arrive in the init records (`theta_deg` next to `pose_world`);
see "Seeding the angle" above.

### Two tests that cannot see this, and why

Both `--theta-residual-ratio` and `--anchor-wrong-surface-margin` were built for this
failure and neither prevents it. The reason is the same for both: they ask where the
moving part should be, and the answer comes from the angle, which is the corrupted
quantity. With the angle wrong, the anchors sat 1.4-1.8 mm from the modelled lid —
the fit believed its own anchors completely. The residual ratio read a healthy 2.0
once settled, and rejection did not fire at all during the 14 frames where the angle
diverged.

So the earlier claim in this file that a *coherently* wrong angle is undetectable was
right, and understated: this failure is exactly that case, and no test whose reference
depends on the estimate can escape it. What worked was not detecting the wrong angle
but not producing it. Both tests are kept — rejection is quiet on healthy captures
(median 0 anchors dropped) and is the only signal that fires on the settled state,
which is worth having as an alarm — but neither should be relied on as a safeguard.

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
