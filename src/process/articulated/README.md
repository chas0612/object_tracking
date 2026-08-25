# Articulated (7-DoF) tracking pipeline

Six degrees of freedom for the static parent plus one joint coordinate, per frame.
The seed accepts both the legacy blue-box `measured` joint file and the promoted
`parts` + `joints[]` mesh schema. Exactly one parent-child joint is required; a
multi-joint file is rejected rather than partially tracked.

**Both joint types are seeded here.** `real_hybrid.py` sweeps the joint coordinate in
whatever unit that joint has — degrees for a hinge, millimetres for a slide — ranks
FoundationPose proposals by multi-view silhouette IoU at each, and writes
`joint_value` in the joint's own units, plus `theta_deg` only when it is genuinely an
angle. The degree-denominated flags (`--theta-min-deg`, `--theta-max-deg`,
`--theta-scan-deg`) are *refused* on a prismatic joint rather than reinterpreted; use
`--joint-min`, `--joint-max`, `--joint-scan` there.

**A prismatic joint's coordinate is ranked on depth, not on the silhouette.** The
silhouette ranking assumes the mask contains the whole object, and a sliding part is
exactly what a mask tends to miss: drawer/2 seeded at frame 270 came back at 0.6 mm
against a truth near 220, because SAM3 given "white box" returned the cabinet without
the extended drawer and a mask holding only the cabinet is best explained by a shut
drawer. Nothing failed; the fit maximised correctly against the wrong target, and the
only tell was the *absolute* IoU — 0.32, against 0.68 on scissors.

`depth_joint.py` scores the moving part's visible pixels against the measured stereo
depth, which is computed over the whole image and does not care what was segmented —
on that frame 200–243k valid pixels per pair against 12–33k inside the mask. It
separates the truth from the shut answer two to one (0.33 at 0 mm, 0.72 at the peak)
where the silhouette spans 0.06 across the entire travel. `--no-joint-depth` restores
the old behaviour; revolute never takes this path and is bit-identical without the
flag (verified by re-running the scissors seed: 6.3812 deg, IoU 0.683850, both).

So the division of labour is **depth decides the joint, the silhouette decides the
body**, and phase 4 alternates the two rather than handing Powell all seven
parameters — with an incomplete mask the joint axis of that search is actively
harmful, since hiding the unmasked part inside the body is what raises the outline
score.

**Seed a sliding joint away from its rest stop.** Shut, this cabinet is very nearly a
symmetric white box — the opening face is flush to 0.7 mm because the drawer fills it
— and FoundationPose's registrations were correctly oriented 26/52 times, a coin
flip, against 45/52 at a frame where the drawer was out. Neither the silhouette
(0.4900–0.5037 across four pairs, the flipped answer winning by 0.008), nor stereo
depth at the handle slots (sign inverted, 8 valid pixels), nor image darkness there
could break the tie. The 180 deg error is not detectable at the rest pose; it is
absent at an open one.

Its signature downstream is a joint clamped at its stop *together with* a bad
residual: 321 of 363 frames at 0 with a mean anchor residual of 11.0 mm, against 7
and 2.9 mm for the correctly-oriented run. The solve keeps asking for negative
displacement because the axis points into the cabinet. Clamping alone is innocent —
see the scissors, closed for most of their take.

**Always pass the extrapolation flag.** `--joint-extrapolate-max` (mesh units) for a
slide, `--theta-extrapolate-max-deg` for a hinge; `run_case.sh` forwards either and
the tracker refuses both at once. Without it `_predict_theta` returns the *previous*
frame's coordinate, so anchors and templates start every frame a full frame of motion
behind. On the blue box that ran the angle away; on drawer/2 it did the opposite and
under-responded, with the joint decaying from its seed while the drawer was actually
shut. Same defect, and the sluggish form is the harder one to see — a still frame
looks aligned, and only the *rate* over the video, or scoring against measured depth,
gives it away. Turning it on took drawer/2 from 205 to 363 frames solved, |dj| p95
from 3.5 to 13.9 mm, and the fit residual from 4.8 to 2.5 mm.

**A sparse depth value must constrain the interval, not merely reset one frame.** A
coherent visual alias can return immediately after a pin: on drawer/2, frame 255 was
pinned to 205 mm and frame 256 jumped straight back to 253.5 mm. Pass
`--joint-anchor-mode trajectory` together with `--joint-anchors`. After GoTrack and
the bidirectional merge, `constrain_joint_trajectory.py` keeps every body pose and
re-solves only the scalar joint trajectory. Stereo values are exact equality
constraints; the default minimum-velocity temporal objective is linear between
them and holds the nearest value outside their span. The raw answer is retained as
`joint_value_before_temporal_constraint`, and the report is written to
`gotrack/joint_trajectory_constraints.json`.

This is deliberately outside MV-GoTrack: it does not change the shared 6-DoF tracker
and it does not pretend that interpolation repairs the underlying correspondence
ambiguity. It turns 25 reliable measurements into constraints on all 363 frames
instead of allowing flow to discard them one frame later.

When 15-frame interpolation is visibly early or late, refine only where the coarse
stereo values prove the joint moved. `make_joint_anchors.py --adaptive-from` compares
adjacent coarse anchors, marks pairs changing by at least 20 mm, pads both ends by one
coarse interval, and measures frames there at stride 5. It does not use GoTrack for
motion detection. On drawer/2 this selects frames 210..315 and needs 14 new stereo
inferences. Existing coarse depth in that window is re-scored against the *current*
run's body poses, rather than mixing joint values measured against an older body
trajectory; coarse values outside the window are reused unchanged.

Dense stereo is still a measurement, not ground truth: on drawer/2 it reported
`210, 230, 210 mm` at frames 260/265/270 although the drawer was visibly against its
upper stop throughout that interval. Use `constrain_joint_trajectory.py
--constraint-mode soft` for this case. Stereo becomes a Huber measurement factor,
acceleration regularises the full trajectory, and `--upper-plateau-tolerance 0.025`
turns contiguous measurements within 25 mm of the observed maximum into a strong
travel-stop plateau. This is opt-in because a freely moving prismatic joint must not
have a stop invented merely because it passed near its largest observed value.

These are scripts, invoked directly, not an importable package — there is no
`__init__.py` and the intra-directory imports are flat, which works because Python
puts a script's own directory on `sys.path`. Run them by path, from anywhere.

## Why FoundationPose here and FoundPose in the 6-DoF pipeline

The 6-DoF pipeline initialises with FoundPose, which lives inside MV-GoTrack and never
touches the FoundationPose repository. FoundPose is template-based and onboards a
template bank per pose configuration — for an articulated object that is one bank per
joint angle, roughly 20 minutes and 1 GB each, which a theta sweep cannot pay for.
FoundationPose renders and compares against the mesh, so a new angle costs a render.
The two are deliberately not unified. See
`docs/offline_capture_pipeline_setup.md` §3 for the setup this implies.

## Stages

| script | what it does |
|---|---|
| `real_depth.py` | stereo depth for one seed frame (TensorRT FoundationStereo) |
| `real_hybrid.py` | the seed: 7-DoF fit, FoundationPose proposals ranked by multi-view silhouette IoU |
| `make_gotrack_init.py` | seed answer → GoTrack's per-camera init JSON, angle included |
| `make_joint_anchors.py` | sparse stereo measurements of a prismatic coordinate |
| `constrain_joint_trajectory.py` | hard stereo constraints → full scalar joint trajectory; body pose unchanged |
| `make_part_init.py` | splits one seed into two independent rigid seeds, body and lid |
| `run_body_bootstrap.sh` | bidirectional rigid GoTrack of the static parent when articulated tracking breaks |
| `compose_body_joint_records.py` | rigid parent poses + sparse joint measurements → articulated records |
| `which_surface.py` | which part's surface each triangulated anchor landed on |
| `theta_trajectory.py` | angle trajectory across solved frames |
| `render_overlay.py` | overlay sheets for a solved frame |
| `render_tracking_video.py` | 2x2 articulated reprojection video from a GoTrack trajectory |
| `run_all_captures.sh` | the whole chain over a list of captures |
| `joint_from_particulate.py` | Particulate `pred.npz` → `joint.json` + part meshes |

`joint_from_particulate.py` undoes the normalisation Particulate applies before
inference (`up_dir` rotation, then a fit into `[-0.5, 0.5]^3`) and which it does not
store. A revolute range is radians and survives that untouched — which is why nothing
here ever had to think about it — but a **prismatic range is a length and must be
multiplied by the mesh's largest bounding-box extent**. Skipping that raises nothing.
`--self-test` round-trips the conversions on a synthetic box, which is the only check
available until a sliding object exists.

Note that `--up-dir` has to be carried across by hand and cannot be inferred: the
up-direction rotations are axis permutations, so the largest extent is invariant under
them and a wrong one changes the axis while leaving every other number intact.

`common.py`, `fit_rc.py`, `probe_fpose.py`, `probe_fpose_theta.py`,
`real_first_pose.py` and `make_synthetic.py` are shared internals.

`which_surface.py` is diagnostic. `make_part_init.py` is also used by the rigid-parent
recovery path below. An articulated failure cannot be attributed from articulated
output alone: tracking the parts independently gives a reference that does not depend
on the joint angle, and that is what made the angle-runaway diagnosis possible.

## Running

```bash
# Dataset-provided articulated seed (tracking-only evaluation). This bypasses
# FoundationPose and stereo depth, but otherwise uses the same direct 7-DoF tracker.
bash src/process/articulated/run_case.sh \
    --capture-dir ~/arctic/workspace/scissor_ep08 \
    --object scissors --seed-frame 10 --run-name arctic_external_seed_full_v1 \
    --mesh ~/arctic/workspace/scissor_mesh/scissors.obj \
    --joint ~/arctic/workspace/scissor_mesh/articulation_particulate/joint.json \
    --external-init-pose ~/arctic/workspace/scissor_ep08/init_pose_world.npy \
    --external-init-joint-value ~/arctic/workspace/scissor_ep08/init_joint_angle_rad.npy \
    --cameras "1 2 3 6 7 8" --direction both --max-frames -1 \
    --input-resize-scale 0.5 --camera-micro-batch-size 2

# One prepared object episode, with an isolated run directory. The capture must
# already contain foundpose_frame_<seed> masks and articulated_probe stereo depth.
bash src/process/articulated/run_case.sh \
    --capture-dir ~/shared_data/capture/test_0810/right/red_bowl/0 \
    --object red_bowl --seed-frame 20 --run-name measured_pivot_v2

# Prismatic second pass: pin the tracker and constrain every output frame between
# sparse stereo measurements. Use a new run name for a clean tracking retry.
bash src/process/articulated/run_case.sh \
    --capture-dir ~/shared_data/capture/test_0810/right/drawer/2 \
    --object drawer_2part --seed-frame 270 --run-name anchored_trajectory_v1 \
    --joint ~/shared_data/capture/test_0810/right/drawer/2/articulated_runs/depth_joint_v2/joint/joint.json \
    --joint-min 0 --joint-max 0.30 --joint-extrapolate-max 0.02 \
    --joint-anchors ~/shared_data/capture/test_0810/right/drawer/2/articulated_runs/joint_anchors_s15.json \
    --joint-anchor-mode trajectory

# The same constraint can be applied independently to an already merged run.
conda run -n object_6d --no-capture-output python src/process/articulated/constrain_joint_trajectory.py \
    --run-dir ~/shared_data/capture/test_0810/right/drawer/2/articulated_runs/anchored_v4/gotrack \
    --object drawer_2part \
    --joint-anchors ~/shared_data/capture/test_0810/right/drawer/2/articulated_runs/joint_anchors_s15.json

# Dense stereo with robust soft factors and a known upper travel stop.
conda run -n object_6d --no-capture-output python \
    src/process/articulated/constrain_joint_trajectory.py \
    --run-dir ~/shared_data/capture/test_0810/right/drawer/2/articulated_runs/anchored_v4/gotrack \
    --object drawer_2part \
    --joint-anchors ~/shared_data/capture/test_0810/right/drawer/2/articulated_runs/joint_anchors_adaptive_s5.json \
    --constraint-mode soft --acceleration-weight 30 --huber-delta 0.01 \
    --upper-plateau-tolerance 0.025 --plateau-weight 30 \
    --joint-min 0 --joint-max 0.30

# Densify only stereo-detected motion, preserving the coarse JSON in a new file.
conda run -n object_6d --no-capture-output python src/process/articulated/make_joint_anchors.py \
    --capture-dir ~/shared_data/capture/test_0810/right/drawer/2 \
    --object drawer_2part \
    --run-dir ~/shared_data/capture/test_0810/right/drawer/2/articulated_runs/anchored_v4/gotrack \
    --adaptive-from ~/shared_data/capture/test_0810/right/drawer/2/articulated_runs/joint_anchors_s15.json \
    --adaptive-stride 5 --adaptive-movement-threshold 0.02 --joint-max 0.30 \
    --out ~/shared_data/capture/test_0810/right/drawer/2/articulated_runs/joint_anchors_adaptive_s5.json

# Recovery when articulated GoTrack loses the trajectory: track only the static
# parent over the full episode, measure the joint against those body poses, then
# compose the two estimates. SE(3) is copied unchanged from rigid GoTrack.
bash src/process/articulated/run_body_bootstrap.sh \
    --capture-dir "$CAPTURE" --source-run "$SOURCE_RUN" \
    --seed-frame "$SEED_FRAME" --run-name body_rigid_v1
conda run -n object_6d --no-capture-output python \
    src/process/articulated/compose_body_joint_records.py \
    --body-run-dir "$CAPTURE/articulated_runs/body_rigid_v1/gotrack" \
    --body-object body_1F --object drawer_2part \
    --joint-anchors "$CAPTURE/articulated_runs/joint_anchors_body_adaptive_s5.json" \
    --out-run-dir "$CAPTURE/articulated_runs/recovered_body_stereo_v1/gotrack"
conda run -n object_6d --no-capture-output python \
    src/process/articulated/constrain_joint_trajectory.py \
    --run-dir "$CAPTURE/articulated_runs/recovered_body_stereo_v1/gotrack" \
    --object drawer_2part \
    --joint-anchors "$CAPTURE/articulated_runs/joint_anchors_body_adaptive_s5.json" \
    --constraint-mode soft --acceleration-weight 30 --huber-delta 0.01 \
    --joint-min 0 --joint-max 0.30

# Safe to poll from another terminal. It reports the current stage and log tail.
python src/process/articulated/case_status.py \
    --capture-dir ~/shared_data/capture/test_0810/right/red_bowl/0 \
    --run-name measured_pivot_v2

# one capture, end to end
SEED_FRAME=40 bash src/process/articulated/run_all_captures.sh 6

# the seed alone
conda run -n object_6d --no-capture-output python src/process/articulated/real_hybrid.py \
    --capture-dir "$CAPTURE" --frames 40 --refine-pairs 2 --budget-gib 6 \
    --theta-max-deg 260
```

All cameras selected in one GoTrack invocation currently need the same decoded
frame dimensions. ARCTIC cameras 1, 2, 3, 6, 7, and 8 are the static 2000x2800
group. Cameras 4 and 5 are static but 2800x2000, so mixing the two groups fails
while materialising the frame bitmap batch; camera 0 is moving and must not be used
with its single schema-compatibility extrinsic.

`run_case.sh` writes only below `<capture>/articulated_runs/<run-name>/` and resumes
completed stages when invoked again. Choose a new run name for a clean retry. The
joint and part meshes are resolved from `~/shared_data/mesh_new/<object>/`; explicit
theta overrides are optional and should be omitted when the published joint limits
are trusted.

## Bidirectional tracking

Tracking carries state from each frame to the next one it visits, so a pass only
ever covers the frames on one side of the frame it picked its seed up on. A forward
run seeded at frame 20 leaves 0-19 untracked; seeded at 250 it abandons 250 frames.

`--direction both`, **the default**, runs the video in each direction from the same
init and merges by frame index. `--frame-order reverse` on the tracker walks the
frames down from the last one; forward deliberately still uses the sequential read
it always had. Merged output lands where a single pass would have written it, so the
video renderer and everything downstream take a merged run without being told.

**Reverse does not seek per frame, and the reason is measured.** Seeking once per
frame was the original implementation and it cost 193 ms per frame across twelve
cameras against 21 ms for the same twelve read sequentially — all-intra MJPG makes a
backward seek correct, not cheap. Worse, the captures live on NFS, where a backward
walk defeats read-ahead outright: drawer/2's reverse pass stalled 5–8 s every twelfth
frame, nineteen times, while its forward pass over the same episode never stalled at
all. `ReverseBlockReader` decodes a short forward block and hands it out backwards,
which took the same sixty frames from 11.5 s to 2.7 s against forward's 1.3 s.
`--reverse-block-size` (default 8) is the block, and it is also the memory: twelve
cameras of 2048x1536 BGR is 113 MB per frame of depth, and 16 measured *slower* than
8 for exactly that reason.

Both passes share one anchor bank: it is a function of the meshes and the joint, not
of the direction, and sharing it makes the two trajectories answers about the same
points. `bidirectional_merge.json` records the per-frame provenance and the
disagreement on any frame both passes solved.

**By default that is one frame — the seed — and it is a seek check, not an accuracy
measure.** Both passes solve it from the same external init pose on the same images,
so agreement there says the reverse pass read the frame it asked for (measured:
1.3e-13 mm on scissors/1) and says nothing about drift. Measuring drift needs
genuine overlap, which needs *two* seeds: seed both an early and a late frame, run
forward from the first and reverse from the second, and every frame between is
solved twice from independent starts with different amounts of accumulated error.
`merge_bidirectional.py` reports that case without changes; only the seeds differ.

The sweep defaults to the full `[min, max]` range in `joint.json`. A signed range
such as red_bowl's `[-140.8, 133.8]` is searched in both directions and always includes
the published zero pose. `--theta-min-deg` and `--theta-max-deg` independently
override those guards when capture evidence shows that a recorded limit is too tight.

## What the seed costs, and where it went

The first-pose stage took 10-18 minutes per episode. Almost none of that was
FoundationPose: measured on red_bowl/1, registration was 84 s and the screening
logic this file is mostly about — global angle selection, quorum, medoid rejection
— was under 1% of the runtime. The rest was the silhouette objective, and 71% of
one evaluation of *that* was a single `np.unique` counting how many distinct pixels
a candidate covered. Sorting 2.1M indices per evaluation was about 60% of the
whole stage.

Two changes, 2026-08-12:

- **The objective no longer sorts.** `--objective-backend numpy` stamps a scratch
  buffer and keeps one representative per pixel — O(n), and **bit-identical** to
  the version it replaced. `torch` scatters into a GPU bitmap, where duplicates
  fold by construction; it is ~110x faster and agrees to 3e-6. `auto` (default)
  takes torch when a CUDA device is visible. Force `numpy` when a reported IoU has
  to be compared digit-for-digit against an older run.
- **The tie-break scans instead of polishing.** It used to run a bounded 7-DoF
  Powell polish per contender, ~600 evaluations each, which between them produced
  exactly one number: which contending grid angle to keep. `--theta-scan-deg`
  sweeps theta with the body frozen at what depth measured — 275 angles for the
  whole range — and yields the shape of the curve rather than three samples of it.
  That matters here because the runners-up are not isolated peaks but a broad band
  on the other side of the joint, which a polish walking into one does not reveal.
  The body still moves in phase 4, which is where it is supposed to.

End to end on red_bowl/1 frame 20: **18 min → 164 s**, of which FoundationPose is
now 84 s. Do not try to shrink that 84 s by sweeping fewer angles. Scoring each
registration's body pose against the full theta curve shows only 2 of 18 angles on
one pair, 5 of 18 on another, and **0 of 18 on the remaining two**, land on a body
pose whose curve peaks at the right angle — the redundancy is the mechanism, not
waste.

**red_bowl has a 180-degree body ambiguity the silhouette cannot resolve, and it is
not a symmetry.** The bowl is close to a solid of revolution, and the two things
projecting from its wall at 90 degrees to the bail are a *lug* and a *spout* --
different parts that look alike at this resolution. So a body rotated 180 degrees
with the handle swung to the other side is a nearly identical silhouette but a
genuinely different pose, and one of the two is wrong. The runs above report -102
and +101.2 degrees, body poses 177.5 degrees apart, IoU within 0.0007. Checked
against the footage, -102 was the wrong one. **Its higher IoU is not why the scan
found the right one.** The polish it replaced improved 1 of 3 contenders and handed
back the other two's input angle unchanged -- the stopping rule documented in
`_polish`, firing -- so it compared one polished candidate against two unpolished
ones and decided by 0.0001. The scan's contribution is that every contender gets
identical treatment, not that it can see the difference: winning margins were 0.004
on ep1 and 0.001 on ep2, both small enough to flip.

Two consequences. Comparing `theta_deg` across runs or episodes means nothing until
the body is put in a canonical frame. And resolving this needs something the
silhouette does not have: the lug/spout asymmetry at full resolution, texture, or a
human. Nothing in this directory does that today.

Environment is `object_6d` for every stage here, and `gotrack` for the tracker that
`run_all_captures.sh` calls at the end.

**`--theta-max-deg` matters.** The joint file's measured range ends at 205.8 deg, which
is how far the *open scan* happened to be opened, not how far the hinge goes — the lid
reaches 213-217 deg. Capping at the scanned value pins every fully-open frame to the
bound where it looks like a measurement. Pass a generous ceiling and treat the limit as
a guard against absurd values, not as knowledge. The real limit is contact with the
table, so it depends on how the body is sitting and is not a constant of the object.

## Machine-specific inputs

Two paths cannot be vendored and are overridable by environment variable:

- `FOUNDATION_STEREO_PLAN` — a TensorRT engine, built for one GPU and one TensorRT
  version. Default is the standalone FoundationStereo checkout.
- `FOUNDATIONPOSE_ROOT` — defaults to the vendored copy at
  `autodex/perception/thirdparty/FoundationPose`, which needs its mycpp extension built
  and its two public checkpoints placed. Set it to a standalone checkout to use one.

The joint file and part meshes are currently read from capture 2's output directory
(`{captures}/2/gotrack_articulated_v2/`), which makes that capture load-bearing for
every other one. `run_all_captures.sh` takes `JOINT` and `PARTS` overrides so moving
them out does not need this directory edited.
