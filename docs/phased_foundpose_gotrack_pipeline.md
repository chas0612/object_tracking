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
isolated SAM3/FoundPose subprocesses rather than keeping those earlier phase
models resident during the GoTrack phase.

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
