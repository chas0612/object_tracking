# Articulated (7-DoF) tracking pipeline

Six degrees of freedom for the body plus one revolute joint angle, per frame. Built
for `blue_plastic_box` (box + hinged lid) but nothing here is specific to it beyond
the joint file.

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
| `make_part_init.py` | splits one seed into two independent rigid seeds, body and lid |
| `which_surface.py` | which part's surface each triangulated anchor landed on |
| `theta_trajectory.py` | angle trajectory across solved frames |
| `render_overlay.py` | overlay sheets for a solved frame |
| `run_all_captures.sh` | the whole chain over a list of captures |

`common.py`, `fit_rc.py`, `probe_fpose.py`, `probe_fpose_theta.py`,
`real_first_pose.py` and `make_synthetic.py` are shared internals.

`make_part_init.py` and `which_surface.py` are diagnostics, not part of the tracking
path. They exist because an articulated failure cannot be attributed from articulated
output alone: tracking the two parts as independent rigid objects gives a reference
that does not depend on the joint angle, and that is what made the angle-runaway
diagnosis possible. Keep them.

## Running

```bash
# one capture, end to end
SEED_FRAME=40 bash src/process/articulated/run_all_captures.sh 6

# the seed alone
conda run -n object_6d python src/process/articulated/real_hybrid.py \
    --capture-dir "$CAPTURE" --frames 40 --refine-pairs 2 --budget-gib 6 \
    --theta-max-deg 260
```

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
