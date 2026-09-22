"""
job_manager.py — Core state machine for a streamer's clip-generation job.

Handles both live-URL recording and direct file uploads through the SAME
pipeline once a raw video file exists, and guarantees the raw video gets
deleted after clips are generated (or if something fails, after a timeout
safety net).

STATES:
  RECORDING  -> actively pulling a live stream
  PROCESSING -> raw video exists (recording stopped, or file was uploaded),
                running auto_clip + vertical_reframe on it
  READY      -> clips are generated and available for download
  FAILED     -> something went wrong (see job.error)

IMPORTANT DESIGN POINT:
  A recording can stop for TWO different reasons, and both must lead to the
  exact same next step (move to PROCESSING):
    1. The user clicks "Stop" -> we send a terminate signal to the process.
    2. The stream ends on its own -> the recording process exits by itself.
  A background watcher thread handles both identically by just waiting on
  the process and reacting to its exit, regardless of why it exited.
"""

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import auto_clip
import proc_utils
import vertical_reframe


def seconds_to_hms(seconds):
    """Convert seconds (float) to yt-dlp/ffmpeg-friendly HH:MM:SS.ms format."""
    seconds = max(0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"

# Only allow this many jobs to be in the CPU-heavy processing stage at once.
# On a fractional-CPU host, two concurrent encodes don't run twice as fast —
# they run twice as slow each, double the peak memory, and make it far more
# likely the container is OOM-killed or fails a health check. Queueing is
# strictly better than thrashing.
MAX_CONCURRENT_PROCESSING = int(os.environ.get("CLIP_MAX_CONCURRENT", "1"))
_processing_slots = threading.Semaphore(MAX_CONCURRENT_PROCESSING)

# Each clip costs a full re-encode, so this directly scales total CPU time.
MAX_CLIPS = int(os.environ.get("CLIP_MAX_CLIPS", "10"))
# YouTube uploads use the segmented strategy (guaranteed spread, not a
# spike count), so each clip is a fully independent download + encode —
# fewer clips means proportionally less total time, which matters more
# here since the whole point of the YouTube fix was reducing wait time.
YOUTUBE_MAX_CLIPS = int(os.environ.get("CLIP_MAX_CLIPS_YOUTUBE", "5"))


def _terminate_process_tree(proc, timeout=10):
    """
    Terminate a subprocess AND any children it spawned. This matters
    specifically for yt-dlp recording a LIVE stream — it commonly shells out
    to ffmpeg as a child process for continuous HLS capture. Calling
    proc.terminate() only kills the yt-dlp parent; the orphaned ffmpeg
    child keeps running (still writing the file — the stream never
    "actually" stops), and because it inherited the same stdout/stderr
    pipes, communicate() in the watcher thread blocks forever waiting for
    those pipes to close — so clips never get generated either. Killing
    the whole process GROUP (not just the one process) fixes both.
    """
    if proc.poll() is not None:
        return  # already exited on its own

    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass

    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    # Didn't die gracefully in time — force kill the whole group so we
    # never end up stuck with an unkillable orphaned recording.
    try:
        if sys.platform == "win32":
            proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def delete_file_with_retry(path, attempts=6, delay=0.5):
    """
    Delete a file robustly. On Windows especially, a file can be transiently
    locked right after ffmpeg finishes writing it — by antivirus scanning it,
    or (very commonly, if the project lives in a synced folder like OneDrive
    or Google Drive) by the sync client grabbing a lock the instant the file
    changes. A single os.remove() attempt can fail for reasons that have
    nothing to do with the deletion logic itself, so retry with backoff
    instead of giving up (or silently swallowing the error) on the first try.
    """
    last_error = None
    for i in range(attempts):
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except (PermissionError, OSError) as e:
            last_error = e
            time.sleep(delay * (i + 1))
    print(f"WARNING: could not delete {path} after {attempts} attempts: {last_error}")
    return False


def generate_thumbnail(clip_path):
    """Grab a frame partway into the clip as a small preview thumbnail."""
    thumb_path = clip_path.replace(".mp4", "_thumb.jpg")
    try:
        proc_utils.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", clip_path, "-ss", "0.5", "-vframes", "1",
                "-vf", "scale=270:-1",
                thumb_path,
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        pass  # a missing thumbnail isn't worth failing the whole job over
    return thumb_path


def find_highlights_segmented(loudness, n_clips=10, min_gap=20.0):
    """
    YouTube-specific highlight strategy: guarantee N clips SPREAD ACROSS
    the whole video, regardless of how the audio behaves.

    find_highlights_with_fallback (used for live Kick streams) looks for
    moments that are LOUDER THAN THE AVERAGE — real spikes against a
    calmer baseline. That fits raw stream mic audio well: reactions,
    laughter, and hype genuinely stand out against quieter talk. It fits
    produced YouTube uploads much less well — podcasts, vlogs, and edited
    videos are typically mastered to a fairly consistent loudness on
    purpose, so there may be few or no real "spikes" to find at all, which
    is how this was collapsing to a single clip (or none) instead of
    failing loudly.

    This function sidesteps that entirely: it divides the video into
    n_clips equal time segments and picks the loudest moment WITHIN each
    segment — a purely relative, local comparison. Because each segment
    is chosen independently by position, this is structurally incapable of
    collapsing to one clip; it always returns up to n_clips highlights,
    each from a different part of the video, however flat the audio is.
    """
    if not loudness:
        return []

    duration = loudness[-1][0]
    if duration <= 0:
        return []

    segment_len = duration / n_clips
    highlights = []
    for i in range(n_clips):
        seg_start = i * segment_len
        seg_end = seg_start + segment_len
        segment_windows = [(t, db) for t, db in loudness if seg_start <= t < seg_end]
        if not segment_windows:
            continue
        peak_t, peak_db = max(segment_windows, key=lambda w: w[1])
        highlights.append((peak_t, 0.0))  # score is informational only here

    return highlights


def find_highlights_with_fallback(loudness, top_n=10, min_gap=20.0):
    """
    Never return empty-handed. Real streams vary a lot in how 'spiky' their
    audio is — a fixed sensitivity threshold that works for one VOD can
    legitimately find nothing in a calmer one. Instead of failing the whole
    job, progressively relax the threshold, and as a last resort just take
    the loudest moments in the recording regardless of how spiky they are.
    """
    for sensitivity in (1.8, 1.4, 1.0, 0.6, 0.3):
        spikes = auto_clip.find_spikes(loudness, sensitivity=sensitivity)
        highlights = auto_clip.group_spikes(spikes, min_gap=min_gap)
        if highlights:
            return highlights

    # Last resort: no meaningful "spikes" at all (very flat audio) — just
    # pick the loudest moments overall, spaced apart, so the streamer still
    # gets clips instead of an error.
    sorted_by_loudness = sorted(loudness, key=lambda x: x[1], reverse=True)
    chosen = []
    for t, db in sorted_by_loudness:
        if all(abs(t - c[0]) > min_gap for c in chosen):
            chosen.append((t, 0.0))
        if len(chosen) >= top_n:
            break
    chosen.sort(key=lambda h: h[0])
    return chosen


class JobStatus(str, Enum):
    RECORDING = "recording"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    jobs_root: str
    status: JobStatus = JobStatus.RECORDING
    created_at: float = field(default_factory=time.time)
    stopped_at: Optional[float] = None
    error: Optional[str] = None
    clips: list = field(default_factory=list)
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    # "live" (Kick/Twitch-style raw stream) uses spike-based detection;
    # "youtube_vod" (produced/mastered uploads) uses segmented coverage.
    # See find_highlights_segmented for why these need different logic.
    source: str = "live"

    @property
    def dir(self):
        return os.path.join(self.jobs_root, self.id)

    @property
    def raw_path(self):
        return os.path.join(self.dir, "raw.mp4")

    @property
    def clips_dir(self):
        return os.path.join(self.dir, "clips")


class JobManager:
    def __init__(self, jobs_root="jobs", raw_video_ttl_seconds=6 * 3600):
        self.jobs_root = jobs_root
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        # Safety net: if a raw video somehow survives (crash, bug, stuck job),
        # a background sweeper deletes anything older than this. Never rely
        # on the happy-path deletion alone for a product handling many users.
        self.raw_video_ttl_seconds = raw_video_ttl_seconds
        threading.Thread(target=self._sweeper_loop, daemon=True).start()

    # ---------- starting a job ----------

    def start_live_job(self, url: str, record_command_builder, source: str = "live") -> Job:
        """
        Start recording a live URL. `record_command_builder(url, out_path)`
        returns the subprocess argv list to run (in production this calls
        yt-dlp; tests can swap in a synthetic ffmpeg source).

        `source` selects which highlight-detection strategy runs later:
        "live" (default) for raw stream audio, "youtube_vod" for produced/
        mastered uploads. See find_highlights_segmented for why.
        """
        job = self._new_job()
        job.source = source
        os.makedirs(job.dir, exist_ok=True)

        cmd = record_command_builder(url, job.raw_path)
        print(f"Starting recording with command: {' '.join(cmd)}")
        # Capture stderr to see what yt-dlp is actually doing.
        # start_new_session=True puts this process (and any children it
        # spawns, e.g. ffmpeg for live HLS) in their own process group, so
        # we can reliably kill ALL of them together when stopping — see
        # _terminate_process_tree for why this matters.
        popen_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        job.process = subprocess.Popen(cmd, **popen_kwargs)

        threading.Thread(target=self._watch_recording, args=(job,), daemon=True).start()
        with self.lock:
            self.jobs[job.id] = job
        return job

    def start_upload_job(self, uploaded_file_path: str) -> Job:
        """A user uploaded a finished VOD directly — skip straight to processing."""
        job = self._new_job()
        os.makedirs(job.dir, exist_ok=True)
        shutil.move(uploaded_file_path, job.raw_path)
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._process_job, args=(job,), daemon=True).start()
        return job

    def start_youtube_vod_job(self, url: str, format_flags: list) -> Job:
        """
        Generate clips from a YouTube video WITHOUT downloading the whole
        thing first. The old path (start_live_job with a full-download
        command) waited for the entire video to download — for a 1-2 hour
        1080p upload, that's most of the processing time before anything
        even starts. This instead:
          1. Downloads ONLY the audio track (a small fraction of the size).
          2. Finds highlight timestamps from that audio alone.
          3. Downloads ONLY the video for those specific highlight windows
             via yt-dlp --download-sections, not the whole file.
        Net effect: total data downloaded and time spent scales with the
        number of clips, not the length of the source video.
        """
        job = self._new_job()
        job.source = "youtube_vod"
        job.status = JobStatus.PROCESSING  # no separate "recording" phase here
        os.makedirs(job.dir, exist_ok=True)
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(
            target=self._process_youtube_vod_job, args=(job, url, format_flags), daemon=True
        ).start()
        return job

    def _process_youtube_vod_job(self, job: Job, url: str, format_flags: list):
        with _processing_slots:
            audio_path = os.path.join(job.dir, "audio.m4a")
            try:
                print(f"Job {job.id}: downloading audio only (fast pre-pass)...")
                audio_cmd = ["yt-dlp", url, "-f", "bestaudio", "-o", audio_path] + format_flags
                result = proc_utils.run(audio_cmd, capture_output=True, text=True, **{})
                if result.returncode != 0 or not os.path.exists(audio_path):
                    raise RuntimeError(f"Could not download audio. yt-dlp error: {result.stderr[-500:]}")

                loudness = auto_clip.measure_loudness(audio_path, window_seconds=1.0)
                print(f"Job {job.id}: audio duration~{loudness[-1][0] if loudness else 0:.0f}s, {len(loudness)} windows")
                highlights = find_highlights_segmented(loudness, n_clips=YOUTUBE_MAX_CLIPS, min_gap=20.0)
                if not highlights:
                    raise RuntimeError("No usable audio found in this video.")

                os.makedirs(job.clips_dir, exist_ok=True)
                print(f"Job {job.id}: downloading {len(highlights)} highlight section(s) only...")
                for i, (t, _z) in enumerate(highlights, start=1):
                    start, end = t - 15, t + 10
                    section = f"*{seconds_to_hms(start)}-{seconds_to_hms(end)}"
                    raw_clip_path = os.path.join(job.clips_dir, f"raw_{i:02d}.mp4")
                    section_cmd = [
                        "yt-dlp", url,
                        "--download-sections", section,
                        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/bestvideo+bestaudio/best",
                        "--merge-output-format", "mp4",
                        "-o", raw_clip_path,
                    ] + format_flags
                    result = proc_utils.run(section_cmd, capture_output=True, text=True)
                    if result.returncode != 0 or not os.path.exists(raw_clip_path):
                        print(f"  [{i}/{len(highlights)}] FAILED to download section: {result.stderr[-300:]}")
                        continue

                    out_path = os.path.join(job.clips_dir, f"clip_{i:02d}_short.mp4")
                    vertical_reframe.process_one(
                        raw_clip_path, out_path, top_ratio=0.5,
                        facecam_box_override=None, no_facecam=False, samples=10,
                    )
                    generate_thumbnail(out_path)
                    delete_file_with_retry(raw_clip_path)  # keep only the final short clip
                    print(f"  [{i}/{len(highlights)}] done")

                job.clips = sorted(f for f in os.listdir(job.clips_dir) if f.endswith("_short.mp4"))
                if not job.clips:
                    raise RuntimeError("Could not generate any clips from this video.")
                job.status = JobStatus.READY

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
            finally:
                delete_file_with_retry(audio_path)

    def _new_job(self) -> Job:
        return Job(id=str(uuid.uuid4())[:8], jobs_root=self.jobs_root)

    # ---------- stopping a recording ----------

    def stop_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.RECORDING:
            return False
        print(f"Job {job_id}: Stop requested by user")
        job.stop_requested = True
        if job.process and job.process.poll() is None:
            print(f"Job {job_id}: Terminating process tree")
            # Run in a thread: _terminate_process_tree can block briefly
            # (up to `timeout` seconds) waiting for a graceful exit before
            # force-killing, and we don't want the API request to hang.
            threading.Thread(target=_terminate_process_tree, args=(job.process,), daemon=True).start()
        return True

    # ---------- the watcher: handles BOTH stop-clicked and stream-ended-naturally ----------

    def _watch_recording(self, job: Job):
        stdout = ""
        stderr = ""
        
        try:
            # Simple approach: wait for the process to finish (either naturally or after terminate)
            # The terminate() call in stop_job() will cause this to exit
            stdout, stderr = job.process.communicate()
                
        except Exception as e:
            print(f"Job {job.id}: Error in watch_recording: {e}")
            if job.process.poll() is None:
                job.process.kill()
                try:
                    stdout, stderr = job.process.communicate(timeout=2)
                except:
                    pass
            
        job.stopped_at = time.time()
        
        # Log the process output for debugging
        print(f"Job {job.id} recording finished with return code: {job.process.returncode}")
        if stdout:
            print(f"Job {job.id} stdout: {stdout}")
        if stderr:
            print(f"Job {job.id} stderr: {stderr}")
            
        # Store stderr for potential error reporting
        job.process_stderr = stderr
        self._process_job(job)

    def _process_job(self, job: Job):
        job.status = JobStatus.PROCESSING
        # Wait for a processing slot before touching ffmpeg. The job shows
        # as PROCESSING while queued, which is honest from the user's side —
        # their clips are coming, just not started yet.
        with _processing_slots:
            self._process_job_inner(job)

    def _process_job_inner(self, job: Job):
        try:
            if not os.path.exists(job.raw_path) or os.path.getsize(job.raw_path) == 0:
                error_msg = "No video was recorded/uploaded (empty or missing file)."
                # Include yt-dlp stderr if available for debugging
                if hasattr(job, 'process_stderr') and job.process_stderr:
                    error_msg += f" yt-dlp error: {job.process_stderr[:500]}"  # Limit error length
                raise RuntimeError(error_msg)

            os.makedirs(job.clips_dir, exist_ok=True)

            # Stage 1: find highlight clips. Which strategy runs depends on
            # the content type — see find_highlights_segmented for why a
            # single approach doesn't fit both raw streams and produced
            # video.
            loudness = auto_clip.measure_loudness(job.raw_path, window_seconds=1.0)
            print(f"Job {job.id}: source={job.source}, duration~{loudness[-1][0] if loudness else 0:.0f}s, {len(loudness)} loudness windows")

            if job.source == "youtube_vod":
                highlights = find_highlights_segmented(loudness, n_clips=YOUTUBE_MAX_CLIPS, min_gap=20.0)
            else:
                highlights = find_highlights_with_fallback(loudness, top_n=MAX_CLIPS, min_gap=20.0)
            # find_highlights_with_fallback only applies top_n on its
            # last-resort path, so enforce the cap here too: take the
            # strongest N, then restore chronological order.
            highlights.sort(key=lambda h: h[1], reverse=True)
            highlights = highlights[:MAX_CLIPS]
            highlights.sort(key=lambda h: h[0])
            print(f"Job {job.id}: cutting {len(highlights)} clip(s)")
            for i, (t, z) in enumerate(highlights, start=1):
                start, end = t - 15, t + 10
                clip_path = os.path.join(job.clips_dir, f"clip_{i:02d}.mp4")
                auto_clip.cut_clip(job.raw_path, start, end, clip_path)

            # Stage 2: reformat every clip to vertical (batch mode), plus a thumbnail for each
            for fname in sorted(os.listdir(job.clips_dir)):
                if fname.endswith(".mp4"):
                    in_path = os.path.join(job.clips_dir, fname)
                    out_path = os.path.join(job.clips_dir, fname.replace(".mp4", "_short.mp4"))
                    vertical_reframe.process_one(
                        in_path, out_path, top_ratio=0.5,
                        facecam_box_override=None, no_facecam=False, samples=10,
                    )
                    generate_thumbnail(out_path)

            job.clips = sorted(
                f for f in os.listdir(job.clips_dir) if f.endswith("_short.mp4")
            )
            job.status = JobStatus.READY

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
        finally:
            # THE KEY REQUIREMENT: delete the raw live/uploaded video once
            # clips are generated (or once we know we failed) — never keep
            # the full recording around longer than needed.
            delete_file_with_retry(job.raw_path)

    # ---------- safety-net cleanup ----------

    def _sweeper_loop(self):
        while True:
            time.sleep(600)  # check every 10 minutes
            now = time.time()
            with self.lock:
                for job in list(self.jobs.values()):
                    if os.path.exists(job.raw_path) and (now - job.created_at) > self.raw_video_ttl_seconds:
                        delete_file_with_retry(job.raw_path)