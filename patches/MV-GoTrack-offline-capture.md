# MV-GoTrack offline-capture patch

## Latest runtime migration (2026-09-30)

The current offline entry points select `MV-GoTrack-next` at
`842bdfeb61f7f15dd649172077cc72ad6a45d718`. The original `MV-GoTrack` checkout is
preserved for `gotrack_capture.py --backend legacy`. Live distributed
`GoTrackEngine`/daemon code and older direct archive scripts retain that legacy
checkout; this migration validates the offline capture pipeline.

The version contract is in [MV-GoTrack-latest.json](MV-GoTrack-latest.json): upstream
SHA, checkpoint digest, the two upstream submodule pins, and patch order. Apply
[MV-GoTrack-latest-capture.patch](MV-GoTrack-latest-capture.patch), then
[MV-GoTrack-latest-articulated.patch](MV-GoTrack-latest-articulated.patch).
The former restores UV-preserving FoundPose onboarding and requires CUDA rendering
while retaining the live renderer's `is_fallback` attribute. The latter adds a
one-joint adapter, articulated bank/projection/renderer/fit helpers, and geometry
tests. Moving-part anchors use alternating articulated tri-fit; rigid IRLS has six
unknowns and must not be used as a seven-parameter solver.

Create a **new** checkout with the helper (it refuses an existing destination and
does not reset the source):

```bash
python scripts/prepare_latest_gotrack.py \
  --source "$HOME/MV-GoTrack" \
  --dependency-source autodex/perception/thirdparty/MV-GoTrack \
  --checkpoint autodex/perception/thirdparty/MV-GoTrack/gotrack_checkpoint.pt
CUDA_VISIBLE_DEVICES=0 conda run -n gotrack python scripts/check_offline_capture_setup.py \
  --component gotrack --require-articulated
```

`--dependency-source` clones the locally available BOP/DINO repositories at the
latest upstream pins. Omit it to use normal submodule initialization (network and
access to those repositories are then required). The checkpoint is linked from
the supplied file after SHA verification; keep that file available. To reproduce
beside an already prepared `MV-GoTrack-next`, specify a different `--destination`.

The complete pre-migration working tree is also captured in
[MV-GoTrack-legacy-runtime.patch](MV-GoTrack-legacy-runtime.patch), against
`a9f033734c0bdf2d191265d22ea732a914c861f6`, excluding submodule pointer changes.
It includes previously unpublished reverse-frame, joint-anchor and body-only
recovery changes. Apply that snapshot by itself to a fresh legacy base; do not
stack the older offline/articulated patches on top of it. Existing old checkouts
are not rewritten by the new helper.

Rigid capture supports `--backend irls` (default) or `trifit`, 128/256 anchors,
one GPU and a full view batch, inline raw-video remapping, and bounded reverse
decoding with original frame indices. Use `legacy` for camera micro-batches or
multi-GPU sharding. The adapter computes the object's diameter from the supplied
mesh; it does not use YCB object-1 metadata or write a mesh-side diameter cache.
The actual bank used by the observer is exported, including the upstream producer
sampler's fields; that rigid export is diagnostic and is not a legacy face-index
bank. Articulated runs continue to share a reusable canonical bank between passes.

On `articulated_tracking`, `src/process/articulated/run_case.sh` selects the latest
bridge by default; `--tracker-backend legacy` restores the archive runner. Existing
FoundationPose/depth seed, merge and scalar trajectory-constraint stages are kept.
Angles are radians internally, sliding displacement is metres, limits and units
are checked, sparse joint anchors are held exactly, and unobserved/rejected joints
retain usable body poses. The earlier `run_all_captures.sh` and
`run_body_bootstrap.sh` are direct legacy entry points.

Validation on this PC (RTX 4080 SUPER, gotrack environment) used existing seeds;
neither the old trajectory nor the projected seed mask is ground truth:

| Case | Conditions | Outcome |
|---|---|---|
| Pink box 6D IRLS | 4 cameras, 256 anchors, resize 0.5, seed frame 30 | 413 frames, no fallback; forward mean step 36.1 ms, p95 37.4 ms |
| Pink box 6D tri-fit | Same capture/seed/cameras | 413 frames, no fallback; forward mean step 35.0 ms, p95 36.7 ms |
| Scissors 7D | 12 cameras, 512 shared anchors, resize 1, existing frame-20 seed | 387 frames, no fallback; forward mean step 220.4 ms, p95 230.5 ms; body and moving-part overlays checked |
| FoundPose | Existing pink-box representation and projected seed mask | Cache loading and GPU candidate inference succeed |
| Articulated geometry | Known revolute/prismatic poses and multiview observations | Joint/pose recovery and unobserved-joint body acceptance pass |
| Articulated integration | Generated six-frame four-camera capture | Compiled observer, both directions, shared bank, JSON export and merge pass |

The primary tests pass (24 tests), along with the articulated branch's five existing
standalone validation scripts and the private articulated fit/bank/prismatic unit
checks. Both latest patches apply cleanly to a fresh upstream base, a prepared
checkout matches the patched files byte-for-byte, GPU preflight passes there, and
a five-frame IRLS capture run succeeds from that reproduced checkout. The legacy
snapshot also applies cleanly to its original base.

Actual local outputs are under:

- Pink box: `capture/eccv2026/robotiq_2f85/box_pink/0/gotrack_migration_20260930/{irls,trifit_v2}`.
- Scissors: `capture/test_0810/right/scissors/0/articulated_runs/latest_migration_20260930_v2`.
- Logs, overlays, generated fixtures and reproduction checks: `outputs/mvgotrack_migration_20260930` in this repository.

The main worktree is based on `tracking-session-progress` (`eaf1336`); the separate
managed worktree `~/.codex/worktrees/mvgotrack-articulated-migration/object_tracking`
is based on `articulated_tracking` (`fc1cabd`). Changes in both are uncommitted;
switching branches in the primary checkout does not transfer the separate worktree's
changes. The original private checkout, `~/MV-GoTrack`, existing result directories
and input meshes/videos were preserved. No commits or pushes were made.

Step timings include GPU synchronization and exclude video decode/remap/resize,
cold compilation, initialization and seed bootstrap. The archived runner does not
export comparable step timings, so these numbers do not establish an exact
speedup ratio. Local legacy/new pose agreement and overlays support compatibility;
accuracy improvement needs independent annotated data. Upstream golden datasets
were unavailable and their regression suite has not been certified here.
Scissors frame 0 has visibly banded/corrupted decoded images and reprojection
misalignment in some cameras in both the existing legacy and latest results;
checked frames 1/5/10/20 do not show that artifact. A successful pose status does
not certify that first frame's image quality. Existing capture files were kept.

## Legacy offline patch

Apply this patch from the root of the approved private MV-GoTrack checkout.
It was generated against commit `a9f033734c0bdf2d191265d22ea732a914c861f6` and contains only the following offline-pipeline changes:

- `archive/run_multiview_gotrack_anchor_online_multi_object.py`: sequential camera micro-batches while keeping all camera observations for one triangulation and rigid fit.
- `utils/renderer_nvdiffrast.py`: require CUDA nvdiffrast instead of silently falling back to the much slower Pyrender renderer.
- `scripts/onboard_custom_mesh_for_foundpose.py`: preserve UV texture maps in an embedded GLB for FoundPose template rendering while retaining the conventional PLY artifact for BOP metadata and compatibility.

## UV-textured FoundPose onboarding

The original onboarding path always rendered templates from a PLY. Because PLY does not carry an OBJ/MTL UV texture map, onboarding first sampled the texture at mesh vertices and stored the result as vertex colors. That approximation is adequate for dense scans, but sparse meshes can lose small labels, logos, and other rotation-breaking appearance evidence.

With this patch, onboarding behaves as follows:

- An input with a valid UV map and material texture is exported as an embedded-texture `obj_XXXXXX.glb`; template RGB/depth/mask rendering uses this GLB.
- A vertex-coloured `obj_XXXXXX.ply` is still written for BOP-compatible geometry and metadata consumers.
- Meshes without a usable UV texture continue to render from PLY, so dense vertex-coloured inputs keep the previous behavior.
- Reusing an existing output chooses the GLB only when it is present; otherwise it falls back to the PLY.
- `summary.json` records `render_model_path`, making the actual template-render input auditable.

This improves the fidelity of the template RGB evidence, but does not guarantee a different final pose. FoundPose uses template appearance to establish DINO correspondences and generate PnP hypotheses; the offline multiview wrapper subsequently performs global silhouette scoring/refinement and uses DINO reprojection only to rerank near-tied mask hypotheses. In a local `instant_rice` A/B study, the UV templates were sharper but the final poses changed by less than `0.6°` and `0.2 mm`, because its dense 253k-vertex PLY already preserved the package texture reasonably well and the silhouette stage converged to the same orientation. The change is expected to matter most for sparse UV meshes such as labelled cans, not for every textured object.

It intentionally does **not** include the local `external/bop_toolkit` or `external/dinov2` submodule pointer changes.

```bash
cd /path/to/MV-GoTrack
git rev-parse HEAD  # must be a9f033734c0bdf2d191265d22ea732a914c861f6
git apply --check /path/to/autodex/patches/MV-GoTrack-offline-capture.patch
git apply /path/to/autodex/patches/MV-GoTrack-offline-capture.patch
```

After applying, run the AutoDex preflight in the `gotrack` environment. Do not apply the legacy `MV-GoTrack-renderer-fix.patch`; it targets an older renderer revision.
