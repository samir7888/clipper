#!/usr/bin/env python3
"""
auto_clip.py — No-AI highlight clipper for stream VODs.

HOW IT WORKS (no AI, no paid APIs, no ML models):
  1. Extracts the audio track from your VOD using ffmpeg.
  2. Measures loudness in short time windows (e.g. every 1 second) using
     ffmpeg's built-in 'astats' filter.
  3. Treats sudden LOUD moments (laughter, screaming, hype, big plays) as
     candidate highlights — a well-known, free heuristic used by manual
     editors for years before "AI highlight detection" was ever sold as
     a product.
  4. Groups nearby loud moments into clip windows with padding before/after.
  5. Cuts each window into its own .mp4 file with ffmpeg, ranked by how
     loud/intense the spike was — so you can review the top N first.

REQUIREMENTS:
  - ffmpeg installed and on PATH (no Python packages required at all;
    this script only uses the Python standard library).

USAGE:
  python3 auto_clip.py path/to/vod.mp4

  Optional flags:
    --outdir DIR         Where to save clips (default: ./clips)
    --top N              Only cut the N loudest highlights (default: 10)
    --min-gap SECONDS    Merge spikes closer than this into one clip (default: 20)
    --pre SECONDS         Seconds of padding before each spike (default: 15)
    --post SECONDS        Seconds of padding after each spike (default: 10)
    --sensitivity FLOAT   Lower = more clips, higher = fewer, stricter clips.
                           This is a z-score threshold above the average
                           loudness of the whole VOD. Default: 1.8

EXAMPLE:
  python3 auto_clip.py my_8hr_stream.mp4 --top 15 --sensitivity 1.5
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys


def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit(
            "ERROR: ffmpeg/ffprobe not found on PATH.\n"
            "Install it (it's free): https://ffmpeg.org/download.html"
        )


def get_duration(video_path):
    """Return duration of the video in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def measure_loudness(video_path, window_seconds=1.0):
    """
    Walk through the audio track in fixed windows and record the mean
    volume (RMS-based, in dB) for each window using ffmpeg's astats filter.

    Returns a list of (start_time_seconds, mean_volume_db) tuples.
    Louder = closer to 0 dB. Quieter = more negative.
    """
    duration = get_duration(video_path)
    results = []
    t = 0.0
    while t < duration:
        seg_len = min(window_seconds, duration - t)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(t), "-t", str(seg_len), "-i", video_path,
            "-af", "astats=metadata=1:reset=1",
            "-f", "null", "-",
        ]
        # astats prints to stderr with -loglevel error unless we ask for info;
        # use a dedicated run that captures stderr with astats output level.
        cmd_info = [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-ss", str(t), "-t", str(seg_len), "-i", video_path,
            "-af", "astats=metadata=1:reset=1",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd_info, capture_output=True, text=True)
        db = parse_rms_db(result.stderr)
        results.append((t, db))
        t += window_seconds
    return results


def parse_rms_db(ffmpeg_stderr):
    """
    Pull 'RMS level dB' out of ffmpeg's astats text output. If parsing
    fails for a window (e.g. total silence), treat it as very quiet.
    """
    for line in ffmpeg_stderr.splitlines():
        line = line.strip()
        if "RMS level dB" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                continue
    return -90.0  # effectively silent / unreadable


def find_spikes(loudness, sensitivity):
    """
    Flag windows that are unusually loud relative to the whole VOD's
    average loudness (a simple z-score threshold). This adapts per-VOD
    so it works whether the streamer runs quiet or hot mic levels.
    """
    values = [db for _, db in loudness if db > -90.0]
    if len(values) < 2:
        return []
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) or 1.0

    spikes = []
    for t, db in loudness:
        z = (db - mean) / stdev
        if z >= sensitivity:
            spikes.append((t, z))
    return spikes


def group_spikes(spikes, min_gap):
    """
    Merge spikes that occur within `min_gap` seconds of each other into
    a single highlight, keeping the timestamp and strength of the loudest
    moment in each group.
    """
    if not spikes:
        return []
    spikes = sorted(spikes, key=lambda s: s[0])
    groups = [[spikes[0]]]
    for t, z in spikes[1:]:
        if t - groups[-1][-1][0] <= min_gap:
            groups[-1].append((t, z))
        else:
            groups.append([(t, z)])

    highlights = []
    for group in groups:
        peak_t, peak_z = max(group, key=lambda s: s[1])
        highlights.append((peak_t, peak_z))
    return highlights


def cut_clip(video_path, start, end, outpath):
    duration = end - start
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(max(0, start)), "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        outpath,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="No-AI VOD highlight clipper (loudness-based).")
    parser.add_argument("video", help="Path to the VOD/video file")
    parser.add_argument("--outdir", default="clips", help="Output folder for clips")
    parser.add_argument("--top", type=int, default=10, help="Max number of clips to cut")
    parser.add_argument("--min-gap", type=float, default=20.0, help="Seconds to merge nearby spikes")
    parser.add_argument("--pre", type=float, default=15.0, help="Padding before spike (seconds)")
    parser.add_argument("--post", type=float, default=10.0, help="Padding after spike (seconds)")
    parser.add_argument("--sensitivity", type=float, default=1.5, help="Z-score threshold for a 'loud moment'")
    parser.add_argument("--window", type=float, default=1.0, help="Audio analysis window size (seconds)")
    args = parser.parse_args()

    check_ffmpeg()

    if not os.path.isfile(args.video):
        sys.exit(f"ERROR: file not found: {args.video}")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Analyzing audio loudness in '{args.video}' (window={args.window}s)...")
    loudness = measure_loudness(args.video, window_seconds=args.window)

    print(f"Scanning for loud moments (sensitivity={args.sensitivity})...")
    spikes = find_spikes(loudness, sensitivity=args.sensitivity)
    highlights = group_spikes(spikes, min_gap=args.min_gap)

    if not highlights:
        print("No highlights found. Try lowering --sensitivity (e.g. 1.2).")
        return

    # Rank by intensity (z-score), take the top N
    highlights.sort(key=lambda h: h[1], reverse=True)
    highlights = highlights[: args.top]
    # Re-sort chronologically for sane filenames/order
    highlights.sort(key=lambda h: h[0])

    print(f"Found {len(highlights)} highlight(s). Cutting clips...")
    manifest = []
    for i, (t, z) in enumerate(highlights, start=1):
        start = t - args.pre
        end = t + args.post
        outpath = os.path.join(args.outdir, f"clip_{i:02d}_t{int(t)}s_score{z:.2f}.mp4")
        cut_clip(args.video, start, end, outpath)
        manifest.append({"clip": outpath, "peak_time_seconds": t, "intensity_score": round(z, 2)})
        print(f"  [{i}/{len(highlights)}] Saved {outpath} (peak at {t:.0f}s, score {z:.2f})")

    manifest_path = os.path.join(args.outdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(highlights)} clip(s) saved to '{args.outdir}/'.")
    print(f"Manifest (timestamps + scores) written to '{manifest_path}'.")


if __name__ == "__main__":
    main()
