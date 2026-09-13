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
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import auto_clip
import vertical_reframe


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
        subprocess.run(
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

    def start_live_job(self, url: str, record_command_builder) -> Job:
        """
        Start recording a live URL. `record_command_builder(url, out_path)`
        returns the subprocess argv list to run (in production this calls
        yt-dlp; tests can swap in a synthetic ffmpeg source).
        """
        job = self._new_job()
        os.makedirs(job.dir, exist_ok=True)

        cmd = record_command_builder(url, job.raw_path)
        print(f"Starting recording with command: {' '.join(cmd)}")
        # Capture stderr to see what yt-dlp is actually doing
        job.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )

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

    def _new_job(self) -> Job:
        return Job(id=str(uuid.uuid4())[:8], jobs_root=self.jobs_root)

    # ---------- stopping a recording ----------

    def stop_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.RECORDING:
            return False
        job.stop_requested = True
        if job.process and job.process.poll() is None:
            job.process.terminate()  # graceful stop signal; recorder finalizes the file
        return True

    # ---------- the watcher: handles BOTH stop-clicked and stream-ended-naturally ----------

    def _watch_recording(self, job: Job):
        stdout, stderr = job.process.communicate()  # blocks until process exits, captures output
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
        try:
            if not os.path.exists(job.raw_path) or os.path.getsize(job.raw_path) == 0:
                error_msg = "No video was recorded/uploaded (empty or missing file)."
                # Include yt-dlp stderr if available for debugging
                if hasattr(job, 'process_stderr') and job.process_stderr:
                    error_msg += f" yt-dlp error: {job.process_stderr[:500]}"  # Limit error length
                raise RuntimeError(error_msg)

            os.makedirs(job.clips_dir, exist_ok=True)

            # Stage 1: find and cut highlight clips (with graceful fallback
            # if the default sensitivity finds nothing on a calmer VOD)
            loudness = auto_clip.measure_loudness(job.raw_path, window_seconds=1.0)
            highlights = find_highlights_with_fallback(loudness, top_n=10, min_gap=20.0)
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
