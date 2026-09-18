# Object-tracking repository cleanup plan

> Status: planning document only. No cleanup or file migration has been performed.
>
> Snapshot date: 2026-09-07  
> Reference branch: `tracking-session-progress`

## 1. Background

This repository started as AutoDex and later became the operational home of the
offline FoundPose + GoTrack object-tracking pipeline. As a result, the current
tracker, old tracking implementations, robot execution/planning code, research
experiments, paper artifacts, and third-party checkouts share one tree.

The repository cannot be cleaned safely by moving or deleting the entire
`autodex/` directory. The current offline tracker still imports a small but
important subset of `autodex/perception/`, `autodex/tracking/`, and one robot
asset below `autodex/planner/`.

The safe direction is therefore:

1. identify the current tracker boundary;
2. extract the still-used AutoDex modules into a tracker-owned package;
3. add tests around the extracted boundary;
4. only then isolate or archive the remaining AutoDex project.

## 2. Current repository classification

### 2.1 Current offline-tracking core

These paths implement or directly orchestrate the pipeline currently used for
capture processing:

- `scripts/distribute_foundpose_onboard.py`
- `scripts/distribute_foundpose_gotrack.py`
- `scripts/distribute_foundpose_gotrack_phased.py`
- `src/process/mask.py`
- `src/process/onboard_foundpose_mesh.py`
- `src/process/foundpose_init_capture.py`
- `src/process/gotrack_capture.py`
- `src/process/run_gotrack_with_inline_undistort.py`
- `src/process/undistort_capture_videos.py`

The current production flow is:

```text
workspace discovery
  -> SAM3 mask at the bootstrap frame
  -> FoundPose candidates and multi-view pose selection/refinement
  -> forward and reverse GoTrack
  -> completeness checks and optional tail recovery
  -> sparse debug sheet / Viser / grid-video inspection
  -> reviewed object_6d_pose_v2.npz promotion
```

The phased NAS-backed scheduler is the primary orchestration path. Older
daemon/session implementations should not be assumed to be part of this path.

### 2.2 Current QA, diagnosis, and promotion tools

The following families are also part of the operational tracker, even though
they do not estimate poses themselves:

- `scripts/prepare_*_tracking_workspace.py`
- `scripts/audit_gotrack_completeness.py`
- `scripts/render_*gotrack*.py`
- `scripts/export_latest_gotrack_poses.py`
- `scripts/export_smoothed_pose_v3.py`
- `scripts/sweep_gotrack_smoothing_gate.py`
- `src/process/render_gotrack_debug_sheet.py`
- `src/process/render_robot_object_reprojection_grid.py`
- `src/process/view_gotrack_viser.py`
- `src/process/view_foundpose_candidates_viser.py`
- `src/process/smooth_gotrack_pose_records.py`
- the object-overlay helpers used by these tools under `src/visualization/`

Campaign selection, retry prompts, rank overrides, promotion holds, and similar
JSON files under `configs/` are operational records. They should be organized,
not indiscriminately deleted.

### 2.3 AutoDex-named modules still required by the current tracker

These paths have AutoDex names but are not yet removable:

- `autodex/perception/mask.py`
- `autodex/perception/foundpose_init.py`
- `autodex/perception/pose_select.py`
- `autodex/perception/silhouette.py`
- `autodex/tracking/inline_undistort.py`
- `autodex/tracking/tail_recovery.py`

There are also narrower dependencies to review:

- `src/process/render_robot_object_reprojection_grid.py` refers to an Inspire
  URDF stored below `autodex/planner/`.
- visualization code imports projection and robot utilities from the separate
  `paradex` repository.
- `autodex/perception/__init__.py` eagerly imports mask, depth, and legacy pose
  modules. A caller that only needs SAM3 can therefore inherit unnecessary
  legacy dependencies.

Before isolating AutoDex, move these modules and assets to a tracker-owned
package and replace broad package imports with direct imports.

### 2.4 Legacy tracking implementations

These paths are tracking-related, but belong to older online/daemon or
FoundationPose-based designs rather than the current phased offline pipeline:

- `autodex/tracking/session.py`
- `autodex/tracking/episode_queue.py`
- `autodex/tracking/progress.py` and the older dashboards where only used by
  the legacy scheduler
- daemon launch/status scripts associated with those implementations
- `src/execution_prev/`
- `src/process/pose.py`
- old batch overlay/onboarding scripts with `~/AutoDex` or
  `~/shared_data/AutoDex` assumptions

`src/process/pose.py` uses the older FoundationPose path. The current initial
pose path uses FoundPose through MV-GoTrack. These similarly named methods must
not be conflated during cleanup.

Legacy status must be confirmed with import/call-site tests before removal.
The first cleanup should move these paths to a clearly labeled legacy area,
not delete them.

### 2.5 AutoDex application code outside object tracking

The following areas are primarily robot task execution, planning, simulation,
grasp generation, or reset logic and are candidates for separation from the
tracking repository:

- `autodex/planner/`
- `autodex/executor/`
- `autodex/simulator/`
- `autodex/visualizer/`
- `src/execution/`
- `src/grasp_generation/`
- `src/reset/`
- `src/scene_generation/`
- planning/simulation portions of `src/validation/`

Some shared assets or small utilities may still be referenced by the tracker.
Those dependencies must be extracted before moving the parent directories.

### 2.6 Research and publication artifacts

These paths should eventually be separated from runtime code, either under a
`research/` namespace or in a dedicated archive repository:

- `Visualization/`
- `rebuttal/`
- `rebuttal_fig.py`
- `rebuttal_fig1.pdf`
- tracked experimental outputs and one-off analysis artifacts
- experiment-specific scripts that do not participate in the maintained
  capture pipeline

### 2.7 Third-party boundaries

Third-party code needs explicit treatment rather than ordinary source-tree
movement:

- `autodex/perception/thirdparty/MV-GoTrack/` is a private, nested Git checkout
  ignored by the top-level repository.
- Required MV-GoTrack changes are preserved in
  `patches/MV-GoTrack-offline-capture.patch` and its description.
- SAM3 is used by the current pipeline.
- the older FoundationPose and `_object_6d_tracking` paths belong primarily to
  older pose pipelines unless a remaining call site proves otherwise.
- the existing `.gitmodules` entries and the manually managed MV-GoTrack
  checkout represent different dependency strategies and should be reconciled.

The cleanup must preserve the known MV-GoTrack base revision, patch application
procedure, external BOP/DINOv2 revisions, checkpoint hash, and environment setup
documented in `docs/offline_capture_pipeline_setup.md`.

## 3. Repository identity problems

The migration from AutoDex to object tracking is incomplete at the repository
metadata level:

- `pyproject.toml` still declares the project name as `autodex` and only finds
  `autodex*` packages.
- `README.md` and `CLAUDE.md` are still primarily AutoDex documents.
- current tracker code is split between `scripts/`, `src/process/`,
  `src/visualization/`, `autodex/perception/`, and `autodex/tracking/`.
- root-level paper and experiment files obscure the supported entry points.

Renaming metadata should happen after the current modules are extracted, not
before, so installation and worker behavior remain reproducible during the
transition.

## 4. Proposed target layout

One possible target is:

```text
tracking_pipeline/
  onboarding/
  segmentation/
  initialization/
  tracking/
  recovery/
  scheduling/
  qa/
  export/

scripts/
  # thin CLI entry points only

configs/
  defaults/       # reusable pipeline defaults
  campaigns/      # episode selections for a campaign
  retries/        # prompt/rank/recovery overrides
  archive/        # completed one-off case-study inputs

assets/
  robot/          # only assets required for tracker QA

docs/
requirements/
patches/
tests/

legacy/
  tracking/       # old online/daemon/FoundationPose paths
  autodex/        # temporary staging before repository separation

research/
  # optional temporary home for paper/experiment artifacts
```

The final outcome may instead use separate repositories for the tracker,
AutoDex application, and publication artifacts. The layout above is useful as
an intermediate, reviewable migration even if a repository split follows.

## 5. Recommended migration sequence

### Phase 0: freeze a reproducible baseline

1. Finish and commit the current uncommitted tracking work.
2. Record the top-level commit, MV-GoTrack base/patch revision, environment
   specifications, and a known successful small run.
3. Tag or branch this state before structural moves.
4. Do not combine behavioral pipeline changes with file relocation.

### Phase 1: add characterization tests

At minimum, cover:

- phased task discovery and state transitions;
- prompt fallback ordering;
- FoundPose cache resolution;
- frame-30 forward/reverse merge behavior;
- GoTrack completeness and tail-recovery acceptance;
- inline undistortion frame/calibration behavior;
- final NPZ frame contract and atomic promotion;
- debug output path resolution.

The repository currently has focused tests for the phased scheduler, inline
undistortion, and pose smoothing, but that is not enough protection for a large
module move.

### Phase 2: extract current tracker internals

1. Create the tracker-owned Python package.
2. Move the required `autodex/perception` and `autodex/tracking` modules without
   changing behavior.
3. Replace `from autodex.perception import ...` with direct imports so unused
   depth and legacy pose dependencies are not loaded.
4. Move the required robot URDF into tracker-owned assets.
5. keep compatibility import shims temporarily if other AutoDex code still
   imports the old paths.

### Phase 3: make CLIs and configuration coherent

1. Convert `scripts/` into thin wrappers around package functions.
2. separate reusable defaults from campaign-specific and retry-only configs;
3. mark completed case-study configs as archived rather than deleting their
   provenance;
4. consolidate overlapping workspace preparation and rendering scripts where
   their data contracts are genuinely identical.

### Phase 4: isolate legacy code

1. Move old daemon/session tracking and FoundationPose paths under `legacy/`.
2. Run the current pipeline smoke tests after each move.
3. Move AutoDex planner/executor/simulator/grasp-generation code only after all
   tracker references have been removed.
4. decide whether `legacy/autodex` remains in this repository or becomes a
   separate archived repository.

### Phase 5: normalize third-party management and project identity

1. choose one explicit dependency strategy for each third-party project;
2. retain the reproducible private MV-GoTrack patch workflow;
3. remove stale submodule declarations only after proving they are unused;
4. update `pyproject.toml`, README, setup documentation, and examples to use
   the object-tracking identity and supported entry points;
5. move or remove root-level research artifacts after preserving them in their
   intended archive.

## 6. Safety rules for the future cleanup

- Never clean a dirty worktree by bulk deletion or relocation.
- Treat top-level Git and the private MV-GoTrack Git checkout as separate
  histories.
- Do not use file existence alone as proof that a legacy path is unused; check
  imports, subprocess command construction, documentation, and worker nodes.
- Preserve campaign manifests, promotion manifests, prompt overrides, and rank
  selections needed to reproduce published pose files.
- Preserve the NAS capture directories as read-only inputs during repository
  restructuring.
- Perform moves in small commits that contain either structural changes or
  behavioral changes, never both.
- Validate on one static/init-only case and one full dynamic GoTrack case before
  deploying a structural change to capture workers.
- Keep compatibility shims until every worker and external consumer has moved
  to the new import/CLI paths.

## 7. First actionable cleanup milestone

The smallest useful future milestone is not deleting AutoDex. It is:

1. introduce a tracker-owned package;
2. move `foundpose_init`, pose selection, silhouette refinement, inline
   undistortion, and tail recovery into it;
3. redirect the phased scheduler and process entry points to those modules;
4. move the required QA robot asset out of the planner tree;
5. pass characterization tests and a one-episode end-to-end smoke run.

After that milestone, the remaining AutoDex and legacy-tracking trees can be
classified and moved with substantially lower risk.
