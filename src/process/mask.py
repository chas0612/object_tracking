#!/usr/bin/env python3
"""Video segmentation (SAM3 or YOLOE).

Supports two modes:
  --capture_dir : process a single episode (all cameras)
  --base        : batch all episodes under a directory (with progress/ETA)

Usage:
    # SAM3 — single episode
    conda activate sam3
    python -u src/process/mask.py \
        --capture_dir /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/apple/20260206_181110 \
        --prompt "object on the checkerboard"

    # SAM3 image model — only frame 0 for FoundPose initialization (low memory)
    conda activate sam3
    python -u src/process/mask.py \
        --capture_dir ~/shared_data/capture/eccv2026/inspire_dftp/apple/0 \
        --frame-index 0 --prompt apple --gpu 0

    # YOLOE — single episode
    conda activate foundationpose
    python -u src/process/mask.py --method yoloe \
        --capture_dir /path/to/episode --conf 0.2

    # SAM3 — batch all episodes
    conda activate sam3
    python -u src/process/mask.py \
        --base /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100

    # Batch with sharding (multi-GPU)
    python -u src/process/mask.py --base ... --shard 0/3 --gpu 0
    python -u src/process/mask.py --base ... --shard 1/3 --gpu 1

    # Filter to specific objects
    python -u src/process/mask.py --base ... --objects apple banana
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

AUTODEX_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUTODEX_ROOT))


# ── Episode discovery ────────────────────────────────────────────────────────

def _find_episodes(base, objects=None):
    """Find all {obj}/{idx} dirs with videos/."""
    base = Path(base)
    episodes = []
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        if objects and obj_dir.name not in objects:
            continue
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir():
                continue
            if (idx_dir / "videos").is_dir():
                episodes.append(idx_dir)
    return episodes


def _is_done(capture_dir):
    """Check if all cameras have mask videos."""
    video_dir = capture_dir / "videos"
    mask_dir = capture_dir / "obj_mask"
    if not mask_dir.exists():
        return False
    video_serials = {p.stem for p in video_dir.glob("*.avi")}
    mask_serials = {p.stem for p in mask_dir.glob("*.avi") if p.stat().st_size > 0}
    return video_serials == mask_serials


# ── Process one episode ──────────────────────────────────────────────────────

def process_episode_sam3(seg, capture_dir, prompt, serials=None):
    """Run SAM3 on all cameras in one episode. Returns (done, failed, total)."""
    from autodex.perception import save_mask_video

    capture_dir = Path(capture_dir)
    video_dir = capture_dir / "videos"
    all_serials = sorted(p.stem for p in video_dir.glob("*.avi"))
    if serials:
        all_serials = [s for s in all_serials if s in set(serials)]

    done = 0
    failed = 0
    skipped = 0

    for cam_idx, serial in enumerate(all_serials):
        mask_path = capture_dir / "obj_mask" / f"{serial}.avi"
        if mask_path.exists() and mask_path.stat().st_size > 0:
            skipped += 1
            continue

        print(f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}.avi", flush=True)
        video_path = str(video_dir / f"{serial}.avi")
        t0 = time.perf_counter()

        masks = seg.segment_video(
            video_path, prompt,
            fallback_prompts=["object"],
        )
        dt = time.perf_counter() - t0

        if masks is None:
            print(f"    {serial}: FAILED ({dt:.1f}s)", flush=True)
            failed += 1
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        save_mask_video(masks, video_path, str(capture_dir), serial, fps, save_debug=True)
        done += 1
        print(f"    {serial}: {len(masks)} frames ({dt:.1f}s)", flush=True)

    return done + skipped, failed, len(all_serials)


def process_episode_yoloe(seg, capture_dir, prompt, batch_size=50, skip=3, serials=None):
    """Run YOLOE on all cameras in one episode. Returns (done, failed, total)."""
    from autodex.perception import save_mask_video

    capture_dir = Path(capture_dir)
    video_dir = capture_dir / "videos"
    all_serials = sorted(p.stem for p in video_dir.glob("*.avi"))
    if serials:
        all_serials = [s for s in all_serials if s in set(serials)]

    done = 0
    failed = 0
    skipped = 0

    for cam_idx, serial in enumerate(all_serials):
        mask_path = capture_dir / "obj_mask" / f"{serial}.avi"
        if mask_path.exists() and mask_path.stat().st_size > 0:
            skipped += 1
            continue

        print(f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}.avi", flush=True)
        video_path = str(video_dir / f"{serial}.avi")
        t0 = time.perf_counter()

        masks = seg.segment_video(
            video_path, prompt, batch_size=batch_size, skip=skip,
        )
        dt = time.perf_counter() - t0

        if masks is None:
            print(f"    {serial}: FAILED ({dt:.1f}s)", flush=True)
            failed += 1
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        save_mask_video(masks, video_path, str(capture_dir), serial, fps, save_debug=True)
        done += 1
        print(f"    {serial}: {len(masks)} frames ({dt:.1f}s)", flush=True)

    return done + skipped, failed, len(all_serials)


def process_episode_sam3_frame(
    seg, capture_dir, prompt, frame_index, output_dir, serials=None, video_dir=None,
    undistort_frame=False,
):
    """Segment one frame from every camera with SAM3's image model.

    This deliberately does not use the SAM3 video predictor.  It is intended
    for FoundPose initialization, which needs one synchronized RGB/mask pair
    per view rather than a mask for every video frame.

    Results are isolated under ``output_dir`` so they never overwrite the
    capture's existing ``obj_mask`` videos or seed-tracking outputs.
    """
    capture_dir = Path(capture_dir)
    video_dir = Path(video_dir).expanduser() if video_dir else capture_dir / "videos"
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    calibration = None
    if undistort_frame:
        calibration_path = capture_dir / "cam_param" / "intrinsics.json"
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"Calibration required by --undistort-frame: {calibration_path}"
            )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    all_serials = sorted(p.stem for p in video_dir.glob("*.avi"))
    if serials:
        wanted = set(serials)
        all_serials = [s for s in all_serials if s in wanted]

    done = failed = skipped = 0
    for cam_idx, serial in enumerate(all_serials):
        image_path = images_dir / f"{serial}.png"
        mask_path = masks_dir / f"{serial}.png"
        if image_path.exists() and mask_path.exists() and mask_path.stat().st_size > 0:
            print(f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}: exists, skip", flush=True)
            skipped += 1
            continue

        video_path = video_dir / f"{serial}.avi"
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError("could not open video")
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_index >= n_frames:
                raise ValueError(f"frame {frame_index} is outside 0..{n_frames - 1}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"could not decode frame {frame_index}")
        except Exception as exc:
            print(f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}: FAILED ({exc})", flush=True)
            failed += 1
            continue
        finally:
            cap.release()

        if calibration is not None:
            entry = calibration.get(serial)
            if not isinstance(entry, dict):
                print(
                    f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}: "
                    "FAILED (missing calibration)", flush=True,
                )
                failed += 1
                continue
            try:
                import numpy as np
                source_k = np.asarray(entry["original_intrinsics"], dtype=np.float64).reshape(3, 3)
                target_k = np.asarray(entry["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
                distortion = np.asarray(entry.get("dist_params", []), dtype=np.float64).reshape(-1)
                bgr = cv2.undistort(bgr, source_k, distortion, None, target_k)
            except (KeyError, TypeError, ValueError) as exc:
                print(
                    f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}: "
                    f"FAILED (bad calibration: {exc})", flush=True,
                )
                failed += 1
                continue

        cv2.imwrite(str(image_path), bgr)
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = seg.segment(rgb, prompt)
        dt = time.perf_counter() - t0
        if mask is None or not mask.any():
            # Keep the RGB input for diagnosis, but do not write a fake mask:
            # downstream FoundPose can use only the cameras that succeeded.
            print(f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}: no mask ({dt:.2f}s)", flush=True)
            failed += 1
            continue

        cv2.imwrite(str(mask_path), mask)
        done += 1
        print(
            f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}: "
            f"{int((mask > 0).sum())} px ({dt:.2f}s)",
            flush=True,
        )

    metadata = {
        "source_capture_dir": str(capture_dir.resolve()),
        "source_video_dir": str(video_dir.resolve()),
        "frame_index": int(frame_index),
        "prompt": prompt,
        "method": "sam3_image",
        "undistort_frame": bool(undistort_frame),
        "serials_requested": all_serials,
        "masks_written": int(done),
        "masks_skipped": int(skipped),
        "masks_failed": int(failed),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return done + skipped, failed, len(all_serials)


def process_static_images_sam3(
    seg, capture_dir, image_dir, prompt, output_dir, serials=None,
):
    """Undistort and segment one already-decoded PNG per camera.

    Both RGB inputs and masks are written below ``output_dir``.  The source
    capture is read-only, which is important for final/static datasets.
    """
    capture_dir = Path(capture_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    with open(capture_dir / "cam_param" / "intrinsics.json", encoding="utf-8") as f:
        calibration = json.load(f)
    all_paths = sorted(image_dir.glob("*.png"))
    if serials:
        wanted = set(serials)
        all_paths = [path for path in all_paths if path.stem in wanted]

    done = failed = skipped = 0
    for cam_idx, source_path in enumerate(all_paths):
        serial = source_path.stem
        image_path = images_dir / source_path.name
        mask_path = masks_dir / source_path.name
        if image_path.is_file() and mask_path.is_file() and mask_path.stat().st_size > 0:
            skipped += 1
            continue
        bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        entry = calibration.get(serial)
        if bgr is None or not isinstance(entry, dict):
            print(f"  cam [{cam_idx+1}/{len(all_paths)}] {serial}: missing image/calibration", flush=True)
            failed += 1
            continue
        try:
            import numpy as np
            source_k = np.asarray(entry["original_intrinsics"], dtype=np.float64).reshape(3, 3)
            target_k = np.asarray(entry["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
            distortion = np.asarray(entry.get("dist_params", []), dtype=np.float64).reshape(-1)
            bgr = cv2.undistort(bgr, source_k, distortion, None, target_k)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  cam [{cam_idx+1}/{len(all_paths)}] {serial}: bad calibration ({exc})", flush=True)
            failed += 1
            continue
        cv2.imwrite(str(image_path), bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        mask = seg.segment(rgb, prompt)
        dt = time.perf_counter() - t0
        if mask is None or not mask.any():
            print(f"  cam [{cam_idx+1}/{len(all_paths)}] {serial}: no mask ({dt:.2f}s)", flush=True)
            failed += 1
            continue
        cv2.imwrite(str(mask_path), mask)
        done += 1
        print(f"  cam [{cam_idx+1}/{len(all_paths)}] {serial}: {int((mask > 0).sum())} px ({dt:.2f}s)", flush=True)

    metadata = {
        "source_capture_dir": str(capture_dir.resolve()),
        "source_image_dir": str(image_dir.resolve()),
        "prompt": prompt,
        "method": "sam3_static_image",
        "serials_requested": [path.stem for path in all_paths],
        "masks_written": done,
        "masks_skipped": skipped,
        "masks_failed": failed,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return done + skipped, failed, len(all_paths)


# ── Format time ──────────────────────────────────────────────────────────────

def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Video segmentation (SAM3 or YOLOE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes:
  --capture_dir DIR    Process a single episode (all cameras)
  --base DIR           Batch all episodes under DIR (with progress/ETA)

examples:
  %(prog)s --capture_dir /path/to/apple/20260206_181110
  %(prog)s --method yoloe --capture_dir /path/to/episode --conf 0.2
  %(prog)s --base /path/to/selected_100 --shard 0/3 --gpu 0
  %(prog)s --base /path/to/selected_100 --objects apple banana
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture_dir", type=str, help="Single episode directory")
    group.add_argument("--base", type=str, help="Batch: parent of all episodes")

    parser.add_argument("--method", choices=["sam3", "yoloe"], default="sam3")
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--serials", nargs="+", default=None,
                        help="Only process these camera serials")
    parser.add_argument(
        "--frame-index", type=int, default=None,
        help=("Run SAM3 image segmentation on only this video frame instead of "
              "propagating masks through every frame. Intended for FoundPose init."),
    )
    parser.add_argument(
        "--frame-output-dir", type=str, default=None,
        help=("Output directory for --frame-index. Default: "
              "<capture_dir>/foundpose_frame_<index:06d>/. Contains images/ and masks/."),
    )
    parser.add_argument(
        "--video-dir", type=str, default=None,
        help=("Video directory for --frame-index. Default: <capture_dir>/videos. "
              "Use <capture_dir>/undistorted_video for calibration-consistent FoundPose inputs."),
    )
    parser.add_argument(
        "--undistort-frame", action="store_true",
        help=("With --frame-index, undistort the decoded source frame using "
              "cam_param/intrinsics.json before SAM3 and FoundPose."),
    )
    parser.add_argument(
        "--static-image-dir", type=str, default=None,
        help=("Read one PNG per camera from this directory, undistort it using "
              "<capture_dir>/cam_param/intrinsics.json, and write images/masks "
              "under --frame-output-dir."),
    )

    # Batch-mode options
    parser.add_argument("--objects", nargs="*", default=None,
                        help="Only process these object names (batch mode)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard spec: RANK/TOTAL (e.g. 0/3)")

    # YOLOE-specific
    parser.add_argument("--conf", type=float, default=0.2,
                        help="YOLOE confidence threshold (default: 0.2)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="YOLOE batch size (default: 50)")
    parser.add_argument("--skip", type=int, default=3,
                        help="YOLOE frame skip (default: 3)")
    args = parser.parse_args()

    if args.static_image_dir is not None:
        if args.capture_dir is None or args.frame_output_dir is None:
            parser.error("--static-image-dir requires --capture_dir and --frame-output-dir")
        if args.method != "sam3":
            parser.error("--static-image-dir currently supports only --method sam3")
    if args.frame_index is not None:
        if args.capture_dir is None:
            parser.error("--frame-index requires --capture_dir (batch mode is not supported)")
        if args.frame_index < 0:
            parser.error("--frame-index must be non-negative")
        if args.method != "sam3":
            parser.error("--frame-index currently supports only --method sam3")
    elif args.undistort_frame:
        parser.error("--undistort-frame requires --frame-index")

    # Load segmentor
    if args.method == "sam3":
        if args.frame_index is None and args.static_image_dir is None:
            from autodex.perception import Sam3Segmentor
            print(f"Loading SAM3 video model on GPU {args.gpu}...", flush=True)
            seg = Sam3Segmentor(gpu=args.gpu)
        else:
            from autodex.perception import Sam3ImageSegmentor
            print(f"Loading SAM3 image model on GPU {args.gpu}...", flush=True)
            seg = Sam3ImageSegmentor(gpu=args.gpu)
    else:
        from autodex.perception import YoloeSegmentor
        print(f"Loading YOLOE on GPU {args.gpu} (conf={args.conf})...", flush=True)
        seg = YoloeSegmentor(gpu=args.gpu, conf_thr=args.conf)
    print("Segmentor ready.", flush=True)

    def _run_episode(capture_dir):
        if args.static_image_dir is not None:
            return process_static_images_sam3(
                seg, capture_dir, args.static_image_dir, args.prompt,
                args.frame_output_dir, args.serials,
            )
        if args.method == "sam3":
            return process_episode_sam3(seg, capture_dir, args.prompt, serials=args.serials)
        else:
            return process_episode_yoloe(seg, capture_dir, args.prompt,
                                         batch_size=args.batch_size, skip=args.skip,
                                         serials=args.serials)

    if args.capture_dir:
        # Single episode mode
        if args.frame_index is None:
            done, failed, total = _run_episode(args.capture_dir)
        else:
            output_dir = (
                Path(args.frame_output_dir).expanduser()
                if args.frame_output_dir
                else Path(args.capture_dir).expanduser()
                / f"foundpose_frame_{args.frame_index:06d}"
            )
            print(f"Single-frame SAM3: frame={args.frame_index}, output={output_dir}", flush=True)
            done, failed, total = process_episode_sam3_frame(
                seg=seg,
                capture_dir=args.capture_dir,
                prompt=args.prompt,
                frame_index=args.frame_index,
                output_dir=output_dir,
                serials=args.serials,
                video_dir=args.video_dir,
                undistort_frame=args.undistort_frame,
            )
        print(f"\nDone! {done}/{total} cameras, {failed} failed.", flush=True)
    else:
        # Batch mode
        episodes = _find_episodes(args.base, args.objects)

        # Sharding
        if args.shard:
            rank, n_shards = map(int, args.shard.split("/"))
            episodes = [e for i, e in enumerate(episodes) if i % n_shards == rank]
            print(f"Shard {rank}/{n_shards}: {len(episodes)} episodes", flush=True)

        todo = [e for e in episodes if not _is_done(e)]
        print(f"Total: {len(episodes)} episodes, {len(episodes) - len(todo)} done, "
              f"{len(todo)} to process", flush=True)

        if not todo:
            print("Nothing to do.")
            return

        total_done = 0
        total_failed = 0
        t_start = time.perf_counter()
        base = Path(args.base)

        for i, capture_dir in enumerate(todo):
            rel = capture_dir.relative_to(base)
            elapsed = time.perf_counter() - t_start

            if i > 0:
                avg = elapsed / i
                remaining = avg * (len(todo) - i)
                eta_str = f"ETA {_format_time(remaining)}"
            else:
                eta_str = "ETA --"

            # Re-check in case another process finished it
            if _is_done(capture_dir):
                print(f"[{i+1}/{len(todo)}] {rel} SKIP (done)", flush=True)
                continue

            print(f"\n[{i+1}/{len(todo)}] {rel}  "
                  f"(elapsed {_format_time(elapsed)}, {eta_str})", flush=True)

            try:
                done, failed, total_cams = _run_episode(capture_dir)
                total_done += done
                total_failed += failed
                print(f"  {done}/{total_cams} cameras, {failed} failed", flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

        total_elapsed = time.perf_counter() - t_start
        print(f"\nAll done! {total_done} cameras processed, {total_failed} failed, "
              f"{_format_time(total_elapsed)} total", flush=True)


if __name__ == "__main__":
    main()
