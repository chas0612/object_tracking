#!/usr/bin/env bash
# The whole articulated chain on every box_articulated capture except 2.
#
# Capture 2 is the one the method was built against and is deliberately excluded:
# it cannot test anything, having been used to choose every threshold in the
# pipeline. These are the frames that can.
#
# Per capture: undistort -> SAM3 mask at one frame -> stereo depth -> articulated
# silhouette fit (7-DoF) -> GoTrack init -> articulated GoTrack over every frame.
#
# The seed frame is a guess. There is no way to pick it without looking at the
# footage, and a frame where the hand covers the lid will produce an answer the
# pipeline itself flags as disputed rather than a silent failure -- which is the
# point of that flag.
set -u
ROOT=~/shared_data/capture/eccv2026/capture_hand/right/box_articulated
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$REPO/src/process/articulated"
GT=$REPO/autodex/perception/thirdparty/MV-GoTrack
JOINT=${JOINT:-$ROOT/2/gotrack_articulated_v2/joint_relaxed.json}
PARTS=${PARTS:-$ROOT/2/gotrack_articulated_v2}
SEED_FRAME=${SEED_FRAME:-40}
PAIRS="22684755:23263780 22645021:23180202 25452066:26256735 22645026:22645029"
CAMS="22641005 22641023 22645021 22684737 22684755 23012641 23028333 23263780 23280286 23280594 25452066 26256735"

for IDX in "$@"; do
    D=$ROOT/$IDX
    F=$(printf %06d "$SEED_FRAME")
    echo "================ capture $IDX (seed frame $SEED_FRAME)"

    if [ "$(ls "$D"/undistorted_video/*.avi 2>/dev/null | wc -l)" -lt 20 ]; then
        echo "-- undistort"
        (cd "$REPO" && conda run -n object_6d --no-capture-output python -u \
            src/process/undistort_capture_videos.py --capture-dir "$D") \
            || { echo "!! undistort failed for $IDX"; continue; }
    fi

    if [ ! -f "$D/foundpose_frame_$F/metadata.json" ]; then
        echo "-- mask"
        (cd "$REPO" && conda run -n sam3 --no-capture-output python -u src/process/mask.py \
            --capture_dir "$D" --frame-index "$SEED_FRAME" --prompt "blue plastic box" \
            --video-dir "$D/undistorted_video" --gpu 0) \
            || { echo "!! mask failed for $IDX"; continue; }
    fi

    if [ ! -f "$D/articulated_probe/frame_$F/depth/depth_manifest.json" ]; then
        echo "-- depth"
        (cd "$HERE" && conda run -n object_6d --no-capture-output python -u \
            real_depth.py --capture-dir "$D" --frame-index "$SEED_FRAME" --pairs $PAIRS) \
            || { echo "!! depth failed for $IDX"; continue; }
    fi

    if [ ! -f "$D/articulated_probe/frame_$F/hybrid/hybrid_result.json" ]; then
        echo "-- articulated silhouette fit"
        (cd "$HERE" && conda run -n object_6d --no-capture-output python -u \
            real_hybrid.py --capture-dir "$D" --frames "$SEED_FRAME" \
            --refine-pairs 2 --budget-gib 6 --theta-max-deg 260) \
            || { echo "!! hybrid failed for $IDX"; continue; }
    fi

    echo "-- gotrack init"
    (cd "$HERE" && conda run -n object_6d --no-capture-output python -u \
        make_gotrack_init.py --capture-dir "$D" --frame-index "$SEED_FRAME" \
        --repeat-frames "$((SEED_FRAME + 4))") \
        || { echo "!! init failed for $IDX"; continue; }

    echo "-- articulated gotrack"
    OUT=$D/gotrack_articulated
    rm -rf "$OUT"
    (cd "$GT" && PYTHONPATH=. PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
        conda run -n gotrack --no-capture-output python \
        archive/run_multiview_gotrack_anchor_online_multi_object.py \
        --input-root "$D" --output-root "$OUT" \
        --checkpoint-path gotrack_checkpoint.pt --template-renderer-backend nvdiffrast \
        --object-names blue_plastic_box --object-ids 1 \
        --mesh-paths "$PARTS/dec_body.obj" \
        --init-pose-sources "$D/gotrack_init/frame_$F" \
        --anchor-bank-paths "$OUT/anchor_bank.npz" \
        --articulation-json "$JOINT" --mask-free --num-anchors 512 --init-min-score 0.0 \
        --num-iters 1 --skip-pnp \
        --optimized-input-pipeline-v2 --optim-v2-crop-camera-workers 4 \
        --optim-v2-warp-grid-workers 4 --optim-template-update-interval 2 \
        --forward-precision fp32 --torch-compile off \
        --worker-mode auto --tri-fit-worker-mode process --triangulation-worker-mode auto \
        --status-log-every 50 --debug-level 0 --camera-ids $CAMS) \
        || echo "!! gotrack failed for $IDX"
    echo "== capture $IDX done"
done
echo "ALL CAPTURES DONE"
