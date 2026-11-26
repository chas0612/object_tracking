# Phased persistent FoundPose + GoTrack pipeline

`scripts/distribute_foundpose_gotrack_phased.py` is an alternative scheduler
for future campaigns. It does not read or modify legacy scheduler state. A
schedule has four durable barriers:

1. `mask`: decode and undistort only the init frame into a lossless PNG, then
   use one persistent SAM3 image model per worker;
2. `foundpose`: one persistent DINO model per worker, with tasks grouped by
   object so its representation and silhouette renderer are reused;
3. `gotrack`: decode raw video and remap each frame in memory, with no
   intermediate undistorted AVI, in one isolated subprocess per episode;
4. `debug`: compact reprojection sheet generation and central publication.

Each phase must complete before the next is launched. Status and attempts are
recorded independently for every task and phase. A stopped worker can be
released with `--mode reset-running --phase PHASE --confirm-workers-stopped`.
Failed phase tasks are retried by relaunching that phase with `--retry-failed`.
By default, a final failure is marked `skipped` in every downstream phase so
unrelated episodes continue through the complete pipeline.

## Example

Create a queue (paths and worker addresses are intentionally placeholders):

```bash
python -u scripts/distribute_foundpose_gotrack_phased.py \
  --mode init \
  --schedule-id CAMPAIGN_gotrack_phased_01 \
  --target-root-rel object_tracking/campaigns/CAMPAIGN/workspace \
  --runs-root-rel object_tracking/campaigns/CAMPAIGN/phased_runs \
  --num-cameras 22 \
  --camera-micro-batch-size 22 \
  --foundpose-selection-mode global \
  --foundpose-global-asymmetry-force \
  --foundpose-global-dino-rerank \
  --debug-sheet-output-root-rel object_tracking/campaigns/CAMPAIGN/debug_sheets
```

Launch one phase on the same stable worker list each time:

```bash
python -u scripts/distribute_foundpose_gotrack_phased.py \
  --mode launch --phase mask \
  --schedule-id CAMPAIGN_gotrack_phased_01 \
  --runs-root-rel object_tracking/campaigns/CAMPAIGN/phased_runs \
  --workers local user@worker-a user@worker-b
```

Repeat with `--phase foundpose`, `gotrack`, and `debug`. At every FoundPose
launch, the scheduler writes a new assignment for the remaining work. Objects
are greedily balanced by their remaining episode counts, while every episode
of one object stays on the same worker for representation reuse. If a machine
is replaced, stop all workers, reset stale claims, and relaunch with the new
complete worker list; completed episodes are not reassigned.

To drain all four phases from one tmux controller, use `launch-all`. It also
attaches to a phase that is already running, waits for it, and launches only
the next phase. Failed tasks are not retried unless `--retry-failed` is passed.

```bash
python -u scripts/distribute_foundpose_gotrack_phased.py \
  --mode launch-all \
  --schedule-id CAMPAIGN_gotrack_phased_01 \
  --runs-root-rel object_tracking/campaigns/CAMPAIGN/phased_runs \
  --workers local user@worker-a user@worker-b
```

```bash
python -u scripts/distribute_foundpose_gotrack_phased.py \
  --mode status \
  --schedule-id CAMPAIGN_gotrack_phased_01 \
  --runs-root-rel object_tracking/campaigns/CAMPAIGN/phased_runs
```

Primary GoTrack completeness checks and tail recovery retain the legacy
pipeline's thresholds. Tail recovery is exceptional and deliberately uses
isolated SAM3 and initialization subprocesses during the GoTrack phase.

## Stereo initialization for meshes without FoundPose assets

Select `--init-backend stereo_foundationpose` when creating the schedule.
The `foundpose` phase name and `foundpose_init/` output layout are shared with
existing schedules. The phase initializes directly from the metric mesh and
does not require `foundpose_assets`, templates, or DINO. Tail recovery uses the
same initialization backend saved in the schedule manifest.

```bash
python -u scripts/distribute_foundpose_gotrack_phased.py \
  --mode init --schedule-id kist_stereo_blue_box_01 \
  --target-root-rel capture/kist_worldmodel \
  --objects blue_plastic_box --episodes 0 \
  --init-backend stereo_foundationpose
```

Use `launch-all` or the same four phase launches as above. The stereo init
phase defaults to the `object_6d` conda environment and reads the TensorRT
FoundationStereo runtime from `~/object-6d-tracking` on each worker. The
runtime checkout must provide `tracking/core/depth_estimator.py` and a CUDA
compatible TensorRT plan at
`thirdparty/FoundationStereo/pretrained_models/foundation_stereo.plan`.
The initializer uses the `object_tracking` FoundationPose code. It looks for
the two weight sets and a `mycpp` extension matching the worker's Python in
that checkout, then uses the assets in the worker's `~/object-6d-tracking`
installation when needed. The worker does not need extra links inside its repo.
Override worker settings at queue creation with `--stereo-env`,
`--stereo-runtime-root` and `--stereo-engine`. Quote home-relative overrides
such as `'~/object-6d-tracking'` so each worker expands its own home directory.

The episode-by-episode `scripts/distribute_foundpose_gotrack.py` scheduler
accepts the same init options. Create the queue with `--mode init` and
`--init-backend stereo_foundationpose`, then launch it with `--mode launch`
and `--workers ...`. Worker processes read the stereo settings from the queue
manifest. Each episode worker runs both initialization and tracking and needs
both runtimes. With the phased scheduler, separate `--mode launch --phase ...`
commands can assign different lists to `foundpose` and `gotrack` when only some
workers have the pinned `MV-GoTrack-next` checkout and checkpoint.

By default the scheduler considers the four rig stereo pairs. Repeat
`--stereo-pair LEFT RIGHT` to supply a different ordered set of up to four
pairs. A pair needs both videos, both camera calibrations and a SAM3 mask in
the left view. Pairs with different encoded frame counts or FPS are skipped,
since same-index frames cannot be assumed aligned. SAM3, multiview silhouette
refinement and GoTrack may use calibrated cameras outside the stereo pairs.
On KIST, the two uncalibrated ego cameras are excluded and three rig pairs
remain on episodes with matching video timelines.

Each pair contributes one world-frame FoundationPose proposal. The init step
ranks proposals by trimmed multiview silhouette IoU and refines the best
proposal (also the runner-up when the initial robust IoUs differ by at most
0.03). `--stereo-silhouette-iters` controls the iteration count, which defaults
to 60. The selected `init_pose_world.npy`, pre-refinement pose, candidate bank
and pair diagnostics are saved in `foundpose_init/`. Numbered capture folders
can be matched to their mesh and SAM3 prompt with `--object-alias-json`.

## Inline undistortion

The init RGB frame is retained below each attempt's `foundpose_frame_*/images/`
directory because both SAM3 and FoundPose consume it. These are lossless PNGs,
not a re-encoded video. GoTrack reads `videos/*.avi`, caches one fixed-point
remap per camera, and applies it immediately after decode. The debug sheet
renderer follows the same raw-video fallback when `undistorted_video/` is not
present.

## Undistort benchmark

Use `scripts/benchmark_undistort_pipeline.py` to split a representative video
into decode/read, OpenCV remap, and encode/write timings. It writes only to a
temporary local directory. The standalone legacy undistort command remains
available for compatibility and numerical comparisons.
