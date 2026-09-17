#!/usr/bin/env python3
"""
vertical_reframe.py — Turn a landscape clip into a vertical (9:16) Shorts/Reels
layout: gameplay/main footage on top, streamer facecam zoomed in below it.

HOW IT WORKS (no AI/LLM, no paid APIs):
  1. Samples several frames from the clip.
  2. Runs OpenCV's built-in Haar cascade face detector on each sampled frame
     (this ships free inside OpenCV — classical computer vision, not ML/AI,
     and requires no internet call or API key).
  3. Since a streamer's webcam overlay almost always sits in a FIXED spot for
     the whole stream, it clusters the face detections to find that fixed
     region rather than trusting any single frame.
  4. Builds a 1080x1920 vertical video with ffmpeg:
       - TOP portion: the full source frame, cropped-to-fill (no stretching)
       - BOTTOM portion: just the facecam region, cropped and zoomed in
  5. If no face is reliably detected, it tells you and falls back to a
     plain center-crop vertical video instead of guessing wrong.

REQUIREMENTS:
  pip install opencv-python-headless --break-system-packages
  (ffmpeg must already be installed — same as auto_clip.py)

USAGE:
  python3 vertical_reframe.py clip.mp4
  python3 vertical_reframe.py clip.mp4 -o short.mp4 --top-ratio 0.55

  If auto-detection picks the wrong box, override it manually once you know
  the coordinates (x,y,width,height of the facecam box in the ORIGINAL video's
  resolution):
  python3 vertical_reframe.py clip.mp4 --facecam-box 1500,780,420,300

  To just get a plain vertical center-crop with no facecam split:
  python3 vertical_reframe.py clip.mp4 --no-facecam
"""

import argparse
import json
import os
import subprocess
import sys

import proc_utils
from collections import defaultdict

try:
    import cv2
except ImportError:
    sys.exit(
        "ERROR: OpenCV is not installed.\n"
        "Install it (free, one-time): pip install opencv-python-headless --break-system-packages"
    )


def get_video_size(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)["streams"][0]
    return data["width"], data["height"]


def sample_frames(path, n_samples=12):
    """Grab n_samples frames evenly spaced through the video using OpenCV."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(total * i / (n_samples + 1)) for i in range(1, n_samples + 1)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def detect_facecam_box(path, n_samples=12, min_hits=3, debug=False):
    """
    Detect faces across sampled frames and find the most consistent region
    (i.e. the fixed webcam overlay), since a single detection could be noise
    (an NPC face, a poster on screen, etc.) but a FIXED recurring box across
    many frames is almost certainly the real facecam.

    Returns (x, y, w, h) in source pixel coordinates, or None if not confident.
    """
    # Defensive: if OpenCV is installed but broken (a partial/incompatible
    # wheel where cv2 imports yet core classes are missing), don't fail the
    # whole clip job — fall back to the plain vertical crop instead. A
    # streamer would much rather get usable clips without the facecam split
    # than an error and nothing at all.
    try:
        cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
        ]
        cascades = [c for c in cascades if not c.empty()]
        if not cascades:
            print("WARNING: face cascades failed to load; skipping facecam detection.")
            return None
    except AttributeError as e:
        print(f"WARNING: OpenCV is not fully functional ({e}); skipping facecam detection.")
        return None

    frames = sample_frames(path, n_samples=n_samples)
    if not frames:
        return None

    frame_h, frame_w = frames[0].shape[:2]

    # IMPORTANT: minSize must scale with the frame, not be a fixed pixel
    # value. A fixed minSize=(60,60) silently misses any facecam overlay
    # where the actual face is smaller than ~120px — which is a very common
    # size for a compact corner webcam box, not an edge case. 2% of the
    # shorter frame dimension (with a small floor) tracks real face size
    # across both 720p and 1080p+ sources.
    min_size = max(20, int(min(frame_w, frame_h) * 0.02))

    all_boxes = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hits = []
        for cascade in cascades:
            # scaleFactor=1.05 (vs the stricter default 1.1) and
            # minNeighbors=3 (vs 5) trade a bit of CPU for meaningfully
            # better recall on small/angled faces — measured to recover
            # detection on facecams as small as 50px that the stricter
            # defaults missed entirely.
            faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(min_size, min_size))
            for (x, y, w, h) in faces:
                frame_hits.append((x, y, w, h))
        if debug:
            print(f"  [debug] frame hits: {frame_hits}")
        all_boxes.extend(frame_hits)

    if debug:
        print(f"  [debug] total raw detections across {len(frames)} frames: {len(all_boxes)} (min_size={min_size})")

    if len(all_boxes) < min_hits:
        return None

    # Cluster by rough grid cell (quadrant/eighths of frame) to find the
    # region where faces keep reappearing — that's the fixed facecam.
    frame_h, frame_w = frames[0].shape[:2]
    cell_w, cell_h = frame_w / 8, frame_h / 8

    clusters = defaultdict(list)
    for (x, y, w, h) in all_boxes:
        cx, cy = x + w / 2, y + h / 2
        cell = (int(cx // cell_w), int(cy // cell_h))
        clusters[cell].append((x, y, w, h))

    best_cluster = max(clusters.values(), key=len)
    if len(best_cluster) < min_hits:
        return None

    xs = sorted(b[0] for b in best_cluster)
    ys = sorted(b[1] for b in best_cluster)
    ws = sorted(b[2] for b in best_cluster)
    hs = sorted(b[3] for b in best_cluster)

    def median(vals):
        return vals[len(vals) // 2]

    x, y, w, h = median(xs), median(ys), median(ws), median(hs)
    return (x, y, w, h)


def pad_box(x, y, w, h, frame_w, frame_h, pad_ratio=0.9):
    """
    Expand the detected FACE box into a wider FACECAM box — a webcam overlay
    usually shows head + shoulders + some background, not just the face.
    """
    pad_x = int(w * pad_ratio)
    pad_y_top = int(h * pad_ratio)
    pad_y_bottom = int(h * pad_ratio * 1.6)  # more room below for shoulders

    nx = max(0, x - pad_x)
    ny = max(0, y - pad_y_top)
    nx2 = min(frame_w, x + w + pad_x)
    ny2 = min(frame_h, y + h + pad_y_bottom)

    return nx, ny, nx2 - nx, ny2 - ny


def build_split_filter(top_h, bottom_h, facecam_box, out_w=1080):
    """Two-part layout: gameplay on top, zoomed facecam on bottom."""
    fx, fy, fw, fh = facecam_box
    top_filter = (
        f"[0:v]scale={out_w}:{top_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{top_h}[top]"
    )
    bottom_filter = (
        f"[0:v]crop={fw}:{fh}:{fx}:{fy},"
        f"scale={out_w}:{bottom_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{bottom_h}[bottom]"
    )
    return f"{top_filter};{bottom_filter};[top][bottom]vstack=inputs=2[v]"


def build_single_filter(out_h=1920, out_w=1080):
    """
    Fallback layout when no facecam is detected: ONE single full-height
    center-crop of the source filling the whole 1080x1920 frame — no split,
    no duplication.
    """
    return (
        f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h}[v]"
    )


def process_one(input_path, output_path, top_ratio, facecam_box_override, no_facecam, samples, debug=False):
    """Reframe a single clip. Returns the output path on success."""
    src_w, src_h = get_video_size(input_path)
    print(f"  Source resolution: {src_w}x{src_h}")

    facecam_box = None
    if facecam_box_override:
        facecam_box = facecam_box_override
        print(f"  Using manual facecam box: {facecam_box}")
    elif not no_facecam:
        print(f"  Sampling {samples} frames to detect the facecam location...")
        raw_box = detect_facecam_box(input_path, n_samples=samples, debug=debug)
        if raw_box:
            x, y, w, h = pad_box(*raw_box, src_w, src_h)
            facecam_box = (x, y, w, h)
            print(f"  Facecam detected at: x={x}, y={y}, w={w}, h={h}")
        else:
            print("  Could not reliably detect a facecam. Using a single full-height crop instead (no split).")

    # Output size and encoder settings are tunable via environment variables
    # so a low-CPU host (e.g. a 0.1-CPU free tier) can be dialled down
    # without code changes. Encoding cost scales roughly with pixel count,
    # so CLIP_OUT_HEIGHT=1280 (720x1280) is about 2.25x cheaper than
    # 1080x1920 while still being a valid vertical format for Shorts/TikTok.
    out_h = int(os.environ.get("CLIP_OUT_HEIGHT", "1920"))
    out_w = int(out_h * 9 / 16)
    # x264 presets trade CPU for file size. 'veryfast' is a good default;
    # 'ultrafast' cuts CPU dramatically at the cost of a larger file.
    preset = os.environ.get("CLIP_PRESET", "veryfast")
    crf = os.environ.get("CLIP_CRF", "23")
    threads = os.environ.get("CLIP_THREADS", "1")

    if facecam_box:
        top_h = int(out_h * top_ratio)
        bottom_h = out_h - top_h
        filter_complex = build_split_filter(top_h, bottom_h, facecam_box, out_w=out_w)
    else:
        # No facecam found (or --no-facecam was passed): render ONE single
        # full-height crop, not a duplicated top/bottom split.
        filter_complex = build_single_filter(out_h=out_h, out_w=out_w)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", preset, "-crf", crf,
        # Cap threads: on a fractional-CPU host, letting x264 spawn one
        # thread per detected core oversubscribes the CPU quota and can
        # balloon memory, which is a common cause of the container being
        # OOM-killed mid-encode.
        "-threads", threads,
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        output_path,
    ]
    print(f"  Rendering vertical video ({out_w}x{out_h}, preset={preset})...")
    proc_utils.run(cmd, check=True)
    print(f"  Done: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Reframe landscape clip(s) into 9:16 vertical Shorts layout(s).")
    parser.add_argument("input", help="Path to a single clip (.mp4), OR a folder containing multiple clips (e.g. the 'clips' folder auto_clip.py made)")
    parser.add_argument("-o", "--output", default=None, help="Output path for single-file mode (default: <input>_vertical.mp4). Ignored in folder mode.")
    parser.add_argument("--outdir", default=None, help="Output folder for folder mode (default: same folder as input clips)")
    parser.add_argument("--top-ratio", type=float, default=0.5, help="Fraction of the 1920px height given to the top (gameplay) section. Default 0.5")
    parser.add_argument("--facecam-box", type=str, default=None, help="Manual override: x,y,w,h in source pixel coordinates (applied to ALL clips in folder mode)")
    parser.add_argument("--no-facecam", action="store_true", help="Skip face detection; just do a plain vertical crop for both halves")
    parser.add_argument("--samples", type=int, default=12, help="How many frames to sample for face detection")
    parser.add_argument("--debug", action="store_true", help="Print raw face detections per frame (useful when detection isn't finding a facecam that's actually visible)")
    args = parser.parse_args()

    facecam_box_override = None
    if args.facecam_box:
        x, y, w, h = map(int, args.facecam_box.split(","))
        facecam_box_override = (x, y, w, h)

    if os.path.isdir(args.input):
        # FOLDER / BATCH MODE — process every clip, skip ones already reframed
        outdir = args.outdir or args.input
        os.makedirs(outdir, exist_ok=True)

        clip_files = sorted(
            f for f in os.listdir(args.input)
            if f.lower().endswith(".mp4") and "_vertical" not in f.lower() and "_short" not in f.lower()
        )
        if not clip_files:
            sys.exit(f"ERROR: no .mp4 clips found in folder: {args.input}")

        print(f"Found {len(clip_files)} clip(s) to reframe in '{args.input}'.")
        outputs = []
        for i, fname in enumerate(clip_files, start=1):
            in_path = os.path.join(args.input, fname)
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(outdir, f"{base}_short.mp4")
            print(f"\n[{i}/{len(clip_files)}] {fname}")
            try:
                process_one(in_path, out_path, args.top_ratio, facecam_box_override, args.no_facecam, args.samples, args.debug)
                outputs.append(out_path)
            except subprocess.CalledProcessError as e:
                print(f"  FAILED on {fname}: {e}")

        print(f"\nDone. {len(outputs)}/{len(clip_files)} short(s) saved to '{outdir}/'.")

    elif os.path.isfile(args.input):
        # SINGLE FILE MODE
        output = args.output or (os.path.splitext(args.input)[0] + "_vertical.mp4")
        process_one(args.input, output, args.top_ratio, facecam_box_override, args.no_facecam, args.samples, args.debug)

    else:
        sys.exit(f"ERROR: path not found: {args.input}")


if __name__ == "__main__":
    main()