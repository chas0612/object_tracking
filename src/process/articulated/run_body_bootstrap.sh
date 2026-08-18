#!/usr/bin/env bash
# Track only the static parent mesh to recover body poses for sparse joint depth.
set -Eeuo pipefail

usage() {
    echo "Usage: $0 --capture-dir DIR --source-run DIR --seed-frame N --run-name NAME [options]"
    echo "Options: --gpu N --body-object NAME --body-mesh FILE --joint FILE"
    echo "Defaults retain the drawer/body_1F recovery setup."
}

CAPTURE_DIR=""
SOURCE_RUN=""
SEED_FRAME=""
RUN_NAME=""
GPU=0
BODY_OBJECT=body_1F
BODY_MESH="/home/capture16/shared_data/mesh_new/drawer_2part/articulation_particulate/parts/body_1F/body_1F.obj"
JOINT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --capture-dir) CAPTURE_DIR=$2; shift 2 ;;
        --source-run) SOURCE_RUN=$2; shift 2 ;;
        --seed-frame) SEED_FRAME=$2; shift 2 ;;
        --run-name) RUN_NAME=$2; shift 2 ;;
        --gpu) GPU=$2; shift 2 ;;
        --body-object) BODY_OBJECT=$2; shift 2 ;;
        --body-mesh) BODY_MESH=$2; shift 2 ;;
        --joint) JOINT_OVERRIDE=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -z "$CAPTURE_DIR" || -z "$SOURCE_RUN" || -z "$SEED_FRAME" || -z "$RUN_NAME" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Invalid --run-name: $RUN_NAME" >&2
    exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HERE="$REPO/src/process/articulated"
GT="$REPO/autodex/perception/thirdparty/MV-GoTrack"
CONDA_BIN=${CONDA_BIN:-/home/capture16/anaconda3/bin/conda}
OBJECT=$BODY_OBJECT
MESH=$BODY_MESH
if [[ -n "$JOINT_OVERRIDE" ]]; then
    JOINT=$JOINT_OVERRIDE
else
    JOINT="$SOURCE_RUN/joint/joint.json"
fi
FRAME=$(printf '%06d' "$SEED_FRAME")
HYBRID="$SOURCE_RUN/probe/frame_$FRAME/hybrid/hybrid_result.json"
RUN_ROOT="$CAPTURE_DIR/articulated_runs/$RUN_NAME"
INIT_ROOT="$RUN_ROOT/part_init"
INIT="$INIT_ROOT/gotrack_init_$OBJECT/frame_$FRAME"
OUT="$RUN_ROOT/gotrack"
BANK="$RUN_ROOT/anchor_bank.npz"
LOG="$RUN_ROOT/pipeline.log"
CAMERAS=(25452066 22641023 23263775 25452061 23173282 22645026 \
         23280286 23280285 23012641 25452062 23028333 23280594)

for required in "$MESH" "$JOINT" "$HYBRID"; do
    [[ -e "$required" ]] || { echo "Missing prerequisite: $required" >&2; exit 1; }
done
mkdir -p "$RUN_ROOT"
if [[ -e "$RUN_ROOT/completed" ]]; then
    echo "Run already completed: $RUN_ROOT"
    exit 0
fi
exec > >(tee -a "$LOG") 2>&1

if [[ ! -d "$INIT" ]]; then
    echo "[stage] rigid_part_init"
    "$CONDA_BIN" run -n object_6d --no-capture-output python -u \
        "$HERE/make_part_init.py" --capture-dir "$CAPTURE_DIR" \
        --joint-json "$JOINT" --hybrid-result "$HYBRID" \
        --frame-index "$SEED_FRAME" --out-root "$INIT_ROOT"
fi

run_pass() {
    local order=$1 out=$2
    if [[ -f "$out/multi_object_stage_c_summary.json" ]]; then
        echo "[skip] rigid_gotrack ($order): $out"
        return
    fi
    local span
    if [[ "$order" == reverse ]]; then span=(--frame-end "$SEED_FRAME");
    else span=(--frame-begin "$SEED_FRAME"); fi
    echo "[stage] rigid_gotrack ($order)"
    mkdir -p "$out"
    (
        cd "$GT"
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
        "$CONDA_BIN" run -n gotrack --no-capture-output python -u \
            archive/run_multiview_gotrack_anchor_online_multi_object.py \
            --input-root "$CAPTURE_DIR" --output-root "$out" \
            --checkpoint-path gotrack_checkpoint.pt --template-renderer-backend nvdiffrast \
            --object-names "$OBJECT" --object-ids 1 --mesh-paths "$MESH" \
            --init-pose-sources "$INIT" --anchor-bank-paths "$BANK" \
            --mask-free --num-anchors 512 --init-min-score 0.0 --num-iters 1 --skip-pnp \
            --optimized-input-pipeline-v2 --optim-v2-crop-camera-workers 4 \
            --optim-v2-warp-grid-workers 4 --optim-template-update-interval 2 \
            --forward-precision fp32 --torch-compile off --worker-mode auto \
            --tri-fit-worker-mode process --triangulation-worker-mode auto \
            --status-log-every 25 --debug-level 0 --frame-order "$order" \
            "${span[@]}" --max-frames -1 --camera-ids "${CAMERAS[@]}"
    )
}

run_pass forward "$RUN_ROOT/gotrack_forward"
run_pass reverse "$RUN_ROOT/gotrack_reverse"
echo "[stage] merge_bidirectional"
mkdir -p "$OUT"
"$CONDA_BIN" run -n object_6d --no-capture-output python -u \
    "$HERE/merge_bidirectional.py" \
    --forward-dir "$RUN_ROOT/gotrack_forward" \
    --reverse-dir "$RUN_ROOT/gotrack_reverse" \
    --out-dir "$OUT" --object "$OBJECT"
cp "$RUN_ROOT/gotrack_forward/multi_object_stage_c_summary.json" \
   "$OUT/multi_object_stage_c_summary.json"
touch "$RUN_ROOT/completed"
echo "[done] $RUN_ROOT"
