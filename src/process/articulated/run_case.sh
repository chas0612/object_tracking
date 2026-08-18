#!/usr/bin/env bash
# FoundationPose articulated seed -> GoTrack for one prepared capture episode.
set -Eeuo pipefail

usage() {
    echo "Usage: $0 --capture-dir DIR --object NAME --seed-frame N --run-name NAME [options]"
    echo "Options: --gpu N --budget-gib N --max-frames N --theta-min-deg N --theta-max-deg N"
    echo "         --joint-min M --joint-max M   sweep bounds for a PRISMATIC joint, in"
    echo "                        the mesh's units (metres here). The degree-denominated"
    echo "                        flags above are refused on a sliding joint, not"
    echo "                        reinterpreted, and vice versa."
    echo "         --joint FILE   articulation file for the tracker (default: the mesh's)"
    echo "         --direction both|forward|reverse   (default: both)"
    echo "         --cameras all|\"ID ID ...\"   (default: the built-in twelve)"
    echo "         --joint-anchors FILE   depth-measured joint values to pin at named"
    echo "                        frames (make_joint_anchors.py). Second pass only:"
    echo "                        it needs body poses from a completed run."
    echo "         --joint-anchor-mode pin|trajectory   (default: pin). 'trajectory'"
    echo "                        applies the sparse values as exact constraints to all"
    echo "                        output frames after tracking; body poses are untouched."
    echo "         --theta-extrapolate-max-deg N | --joint-extrapolate-max M"
    echo "                        project anchors at the joint coordinate extrapolated"
    echo "                        from the last two frames, capped. Without it every"
    echo "                        frame starts a full frame of motion behind, which"
    echo "                        reads as a joint that under-responds (drawer/2) or"
    echo "                        runs away (blue_plastic_box). One unit or the other."
    echo
    echo "Tracking carries state forward from the frame it picks the seed up on, so a"
    echo "forward pass never covers anything before the seed. 'both' runs the video in"
    echo "each direction from the same init and merges by frame index; the seed frame is"
    echo "solved twice and the gap between those two answers is reported."
    echo
    echo "--theta-min-deg/--theta-max-deg bound the *seed* sweep only. The tracker's"
    echo "clamp comes from range_rad in the articulation file, so widening a limit the"
    echo "trajectory is hitting means passing --joint with an edited copy."
}

CAPTURE_DIR=""
OBJECT=""
SEED_FRAME=""
RUN_NAME=""
GPU=0
BUDGET_GIB=6
MAX_FRAMES=-1
THETA_MIN=""
THETA_MAX=""
JOINT_MIN=""
JOINT_MAX=""
THETA_EXTRAP=""
JOINT_EXTRAP=""
JOINT_ANCHORS=""
JOINT_ANCHOR_MODE=pin
JOINT_OVERRIDE=""
DIRECTION=both
CAMERAS_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --capture-dir) CAPTURE_DIR=$2; shift 2 ;;
        --object) OBJECT=$2; shift 2 ;;
        --seed-frame) SEED_FRAME=$2; shift 2 ;;
        --run-name) RUN_NAME=$2; shift 2 ;;
        --gpu) GPU=$2; shift 2 ;;
        --budget-gib) BUDGET_GIB=$2; shift 2 ;;
        --max-frames) MAX_FRAMES=$2; shift 2 ;;
        --theta-min-deg) THETA_MIN=$2; shift 2 ;;
        --theta-max-deg) THETA_MAX=$2; shift 2 ;;
        --joint-min) JOINT_MIN=$2; shift 2 ;;
        --joint-max) JOINT_MAX=$2; shift 2 ;;
        --theta-extrapolate-max-deg) THETA_EXTRAP=$2; shift 2 ;;
        --joint-extrapolate-max) JOINT_EXTRAP=$2; shift 2 ;;
        --joint-anchors) JOINT_ANCHORS=$2; shift 2 ;;
        --joint-anchor-mode) JOINT_ANCHOR_MODE=$2; shift 2 ;;
        --joint) JOINT_OVERRIDE=$2; shift 2 ;;
        --direction) DIRECTION=$2; shift 2 ;;
        --cameras) CAMERAS_ARG=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$CAPTURE_DIR" || -z "$OBJECT" || -z "$SEED_FRAME" || -z "$RUN_NAME" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "--run-name may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
fi
if [[ "$JOINT_ANCHOR_MODE" != pin && "$JOINT_ANCHOR_MODE" != trajectory ]]; then
    echo "--joint-anchor-mode must be pin or trajectory; got $JOINT_ANCHOR_MODE" >&2
    exit 2
fi
if [[ "$JOINT_ANCHOR_MODE" == trajectory && -z "$JOINT_ANCHORS" ]]; then
    echo "--joint-anchor-mode trajectory requires --joint-anchors FILE" >&2
    exit 2
fi
if [[ -n "$JOINT_ANCHORS" && ! -f "$JOINT_ANCHORS" ]]; then
    echo "No such --joint-anchors file: $JOINT_ANCHORS" >&2
    exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HERE="$REPO/src/process/articulated"
GT="$REPO/autodex/perception/thirdparty/MV-GoTrack"
CONDA_BIN=${CONDA_BIN:-$HOME/anaconda3/bin/conda}
if [[ ! -x "$CONDA_BIN" ]]; then
    CONDA_BIN=$(command -v conda || true)
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
    echo "Cannot find conda; set CONDA_BIN to its executable path" >&2
    exit 1
fi
MESH_ROOT="$HOME/shared_data/mesh_new/$OBJECT"
# Absolute, because the tracker is invoked from inside $GT: a relative --joint would
# resolve against MV-GoTrack's directory rather than the caller's, which fails with a
# path that looks correct in the error message.
if [[ -n "$JOINT_OVERRIDE" ]]; then
    JOINT=$(cd "$(dirname "$JOINT_OVERRIDE")" && pwd)/$(basename "$JOINT_OVERRIDE")
else
    JOINT=$MESH_ROOT/articulation_particulate/joint.json
fi
MESH="$MESH_ROOT/$OBJECT.obj"
FRAME=$(printf '%06d' "$SEED_FRAME")
RUN_ROOT="$CAPTURE_DIR/articulated_runs/$RUN_NAME"
PROBE_ROOT="$RUN_ROOT/probe"
HYBRID="$PROBE_ROOT/frame_$FRAME/hybrid/hybrid_result.json"
INIT="$RUN_ROOT/gotrack_init/frame_$FRAME"
OUT="$RUN_ROOT/gotrack"
LOG_DIR="$RUN_ROOT/logs"
LOG="$LOG_DIR/pipeline.log"

for required in \
    "$JOINT" "$MESH" \
    "$CAPTURE_DIR/articulated_probe/frame_$FRAME/depth/depth_manifest.json" \
    "$CAPTURE_DIR/foundpose_frame_$FRAME/metadata.json"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing prerequisite: $required" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR"
if [[ -e "$RUN_ROOT/completed" ]]; then
    echo "Run already completed: $RUN_ROOT"
    exit 0
fi
if [[ -e "$RUN_ROOT/pipeline.pid" ]]; then
    old_pid=$(<"$RUN_ROOT/pipeline.pid")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Run appears active as PID $old_pid: $RUN_ROOT" >&2
        exit 1
    fi
fi

echo $$ > "$RUN_ROOT/pipeline.pid"
rm -f "$RUN_ROOT/failed"
on_exit() {
    code=$?
    rm -f "$RUN_ROOT/pipeline.pid"
    if [[ $code -ne 0 ]]; then
        touch "$RUN_ROOT/failed"
        echo "[failed] exit=$code"
    fi
}
trap on_exit EXIT
exec > >(tee -a "$LOG") 2>&1

echo "[run] object=$OBJECT capture=$CAPTURE_DIR seed=$SEED_FRAME run=$RUN_NAME"
echo "[run] joint=$JOINT output=$RUN_ROOT gpu=$GPU"

if [[ ! -f "$HYBRID" ]]; then
    echo "[stage] foundationpose_seed"
    theta_args=()
    [[ -n "$THETA_MIN" ]] && theta_args+=(--theta-min-deg "$THETA_MIN")
    [[ -n "$THETA_MAX" ]] && theta_args+=(--theta-max-deg "$THETA_MAX")
    [[ -n "$JOINT_MIN" ]] && theta_args+=(--joint-min "$JOINT_MIN")
    [[ -n "$JOINT_MAX" ]] && theta_args+=(--joint-max "$JOINT_MAX")
    (
        cd "$REPO"
        CUDA_VISIBLE_DEVICES="$GPU" "$CONDA_BIN" run -n object_6d --no-capture-output \
            python -u "$HERE/real_hybrid.py" \
            --capture-dir "$CAPTURE_DIR" --frames "$SEED_FRAME" --object "$OBJECT" \
            --output-root "$PROBE_ROOT" --refine-pairs 2 --budget-gib "$BUDGET_GIB" \
            "${theta_args[@]}"
    )
else
    echo "[skip] foundationpose_seed: $HYBRID"
fi

init_count=0
if [[ -d "$INIT" ]]; then
    init_count=$(find "$INIT" -maxdepth 1 -type f -name '*.json' | wc -l)
fi
if [[ ${init_count:-0} -lt 1 ]]; then
    echo "[stage] gotrack_init"
    "$CONDA_BIN" run -n object_6d --no-capture-output python -u "$HERE/make_gotrack_init.py" \
        --capture-dir "$CAPTURE_DIR" --frame-index "$SEED_FRAME" \
        --hybrid-result "$HYBRID" --out "$INIT"
else
    echo "[skip] gotrack_init: $INIT ($init_count files)"
fi

# The twelve this pipeline was built on. `--cameras all` uses every camera the
# capture has calibration and video for instead, which is what a joint whose motion
# is nearly along the viewing ray needs: a slide is weakly observable head-on and
# strongly observable from an oblique view, and the extra ten here are oblique.
DEFAULT_CAMERAS=(25452066 22641023 23263775 25452061 23173282 22645026 \
                 23280286 23280285 23012641 25452062 23028333 23280594)
if [[ "$CAMERAS_ARG" == all ]]; then
    mapfile -t CAMERAS < <(
        "$CONDA_BIN" run -n object_6d --no-capture-output python - "$CAPTURE_DIR" <<'PYEOF'
import json, os, sys
capture = sys.argv[1]
param = os.path.join(capture, "cam_param")
intrinsics = set(json.load(open(os.path.join(param, "intrinsics.json"))))
extrinsics = set(json.load(open(os.path.join(param, "extrinsics.json"))))
videos = {name[:-4] for name in os.listdir(os.path.join(capture, "undistorted_video"))
          if name.endswith(".avi")}
print("\n".join(sorted(intrinsics & extrinsics & videos)))
PYEOF
    )
    if [[ ${#CAMERAS[@]} -lt 2 ]]; then
        echo "--cameras all found ${#CAMERAS[@]} usable cameras in $CAPTURE_DIR" >&2
        exit 1
    fi
elif [[ -n "$CAMERAS_ARG" ]]; then
    read -r -a CAMERAS <<< "$CAMERAS_ARG"
else
    CAMERAS=("${DEFAULT_CAMERAS[@]}")
fi
echo "[run] cameras=${#CAMERAS[@]} (${CAMERAS[*]})"

# Both passes share one anchor bank. It is a function of the meshes and the joint,
# not of the direction, and sharing it means the two trajectories are answers about
# the same points -- which is what makes their disagreement at the seed frame worth
# reading.
BANK="$RUN_ROOT/anchor_bank.npz"

run_pass() {
    local order=$1 out=$2
    if [[ -f "$out/multi_object_stage_c_summary.json" ]]; then
        echo "[skip] articulated_gotrack ($order): $out/multi_object_stage_c_summary.json"
        return 0
    fi
    # A pass can only track away from the frame it picks the seed up on, so bound it
    # there. Without this the reverse pass seeks and decodes twelve views for every
    # frame after the seed before reaching a single one it can use.
    local span
    if [[ "$order" == reverse ]]; then
        span=(--frame-end "$SEED_FRAME")
    else
        span=(--frame-begin "$SEED_FRAME")
    fi
    local extrap=()
    [[ -n "$THETA_EXTRAP" ]] && extrap+=(--theta-extrapolate-max-deg "$THETA_EXTRAP")
    [[ -n "$JOINT_EXTRAP" ]] && extrap+=(--joint-extrapolate-max "$JOINT_EXTRAP")
    [[ -n "$JOINT_ANCHORS" ]] && extrap+=(--joint-anchor-json "$JOINT_ANCHORS")
    echo "[stage] articulated_gotrack ($order)"
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
            --articulation-json "$JOINT" --mask-free --num-anchors 512 \
            --init-min-score 0.0 --num-iters 1 --skip-pnp \
            --optimized-input-pipeline-v2 --optim-v2-crop-camera-workers 4 \
            --optim-v2-warp-grid-workers 4 --optim-template-update-interval 2 \
            --forward-precision fp32 --torch-compile off \
            --worker-mode auto --tri-fit-worker-mode process \
            --triangulation-worker-mode auto --status-log-every 25 --debug-level 0 \
            --frame-order "$order" "${span[@]}" "${extrap[@]}" \
            --max-frames "$MAX_FRAMES" --camera-ids "${CAMERAS[@]}"
    )
}

case "$DIRECTION" in
    forward)
        run_pass forward "$OUT"
        ;;
    reverse)
        run_pass reverse "$OUT"
        ;;
    both)
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
        ;;
    *)
        echo "--direction must be both, forward or reverse; got $DIRECTION" >&2
        exit 2
        ;;
esac

if [[ "$JOINT_ANCHOR_MODE" == trajectory ]]; then
    echo "[stage] constrain_joint_trajectory"
    "$CONDA_BIN" run -n object_6d --no-capture-output python -u \
        "$HERE/constrain_joint_trajectory.py" \
        --run-dir "$OUT" --object "$OBJECT" --joint-anchors "$JOINT_ANCHORS"
fi

touch "$RUN_ROOT/completed"
rm -f "$RUN_ROOT/failed"
echo "[completed] $RUN_ROOT"
