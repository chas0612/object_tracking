# ECCV human object tracking

The clean source dataset is:

```text
$HOME/shared_data/capture/eccv2026/v0/human
```

FoundPose, SAM3, and GoTrack must not write intermediate results below that
directory. The only approved pipeline output there is a reviewed final
`object_6d_pose_v2.npz`.

## Storage layout

```text
$HOME/shared_data/object_tracking/campaigns/human/
├── workspace/       # relative links to clean inputs; all episode outputs live here
├── onboarding_runs/ # FoundPose onboarding scheduler state and logs
├── runs/            # tracking scheduler state and logs
├── debug_sheets/    # compact review images
├── exports/         # export reports
└── archive/         # superseded reviewed attempts, if needed
```

Canonical meshes and reusable FoundPose representations remain under
`$HOME/shared_data/mesh_new/<object>/`.

## Preflight

Create or validate the write-isolated workspace:

```bash
cd "$HOME/object_tracking"

python scripts/prepare_human_tracking_workspace.py
python scripts/prepare_human_tracking_workspace.py --write

python scripts/audit_human_tracking_inputs.py --allow-missing-cache
```

The workspace selects only the 21 fixed cameras listed in both
`intrinsics.json` and `extrinsics.json`. Two ego videos are intentionally not
linked into pipeline `videos/`.

## Onboarding

Run onboarding only for objects reported as missing cache. Scenario discovery
uses each object's earliest numeric episode containing valid intrinsics, so an
object does not need episode 0.

```bash
python -u scripts/distribute_foundpose_onboard.py \
  --scenario-root-rel capture/eccv2026/v0/human \
  --objects <OBJECTS...> \
  --workers <WORKERS...> \
  --state-root-rel object_tracking/campaigns/human/onboarding_runs \
  --run-name human_onboarding_01 \
  --max-attempts 3
```

Run the audit again without `--allow-missing-cache`; it must report
`ready_for_tracking: true`.

## Tracking queue

Create the queue with a dry run first:

```bash
python scripts/distribute_foundpose_gotrack.py \
  --mode init \
  --dry-run \
  --schedule-id human_gotrack_01 \
  --target-root-rel object_tracking/campaigns/human/workspace \
  --runs-root-rel object_tracking/campaigns/human/runs \
  --debug-sheet-output-root-rel object_tracking/campaigns/human/debug_sheets \
  --num-cameras 21 \
  --camera-micro-batch-size 0
```

Remove `--dry-run` only after the eligible episode count matches the preflight
episode count. The scheduler protects `capture/eccv2026/v0` by default and
will refuse to initialize directly against the clean source tree.

All worker PCs must pull the commit containing these scripts before launch.

## Completeness and final export

```bash
python scripts/audit_gotrack_completeness.py \
  --schedule-id human_gotrack_01 \
  --runs-root-rel object_tracking/campaigns/human/runs
```

After visual review, export only from the reviewed schedule. Human capture
frame counts come from the existing contiguous `object_6d/pose_*.txt` files.

```bash
python scripts/export_latest_gotrack_poses.py \
  --target-root "$HOME/shared_data/capture/eccv2026/v0/human" \
  --shared-root "$HOME/shared_data" \
  --no-latest-manifest \
  --schedule-dir "$HOME/shared_data/object_tracking/campaigns/human/runs/human_gotrack_01" \
  --frame-contract text
```

The command above is a dry run. Add `--write` and an explicit `--report` only
after the reviewed selection and hold list are final.
