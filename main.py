"""
main.py — FastAPI backend for the streamer clip app.

Endpoints:
  POST /jobs/live      {url}   -> start recording a live  URL
  POST /jobs/upload    (file)  -> upload a finished VOD directly
  POST /jobs/{id}/stop         -> manually stop a live recording
  GET  /jobs/{id}                -> poll status + clip list
  GET  /jobs/{id}/clips/{name}   -> download a generated clip

Run locally:
  uvicorn main:app --reload --port 8000
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from job_manager import JobManager, JobStatus

app = FastAPI(title="Streamer Clip Generator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000", 
        "http://127.0.0.1:8000",
        "https://clipper-kgln.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
manager = JobManager(jobs_root=os.path.join(os.path.dirname(__file__), "jobs"))


class LiveRequest(BaseModel):
    url: str


def build_record_command(url: str, out_path: str):
    """
    PRODUCTION command: pull a live stream with yt-dlp until it ends
    naturally or is terminated. Enhanced options for cloud/production reliability.
    """
    return [
        "yt-dlp", url, 
        "-f", "best[height<=720]/best",  # Limit quality to reduce bandwidth/processing
        "-o", out_path, 
        "--no-part",
        "--retries", "10",  # Retry on failures
        "--fragment-retries", "10",  # Retry fragments
        "--socket-timeout", "30",  # Connection timeout
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        "--add-header", "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "--add-header", "Accept-Encoding:gzip, deflate, br",
        "--add-header", "DNT:1",
        "--add-header", "Connection:keep-alive",
        "--add-header", "Upgrade-Insecure-Requests:1",
        "--verbose",  # More detailed logging for debugging
        "--js-runtimes", "deno",  # required since yt-dlp 2025.11.12 for
        # YouTube's signature challenges (see EJS wiki); harmless no-op
        # for Kick. Explicit rather than relying on auto-detection so a
        # future yt-dlp default change can't silently disable it.
    ]


@app.post("/jobs/live")
def start_live(req: LiveRequest):
    job = manager.start_live_job(req.url, build_record_command)
    return {"job_id": job.id, "status": job.status}


@app.post("/jobs/upload")
def upload_vod(file: UploadFile = File(...)):
    tmp_path = tempfile.mktemp(suffix=".mp4")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job = manager.start_upload_job(tmp_path)
    return {"job_id": job.id, "status": job.status}


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    ok = manager.stop_job(job_id)
    if not ok:
        raise HTTPException(404, "Job not found or not currently recording")
    return {"job_id": job_id, "status": "stopping"}


@app.get("/jobs/{job_id}")
def get_status(job_id: str):
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "error": job.error,
        "clips": job.clips,
    }


@app.get("/jobs/{job_id}/clips/{clip_name}")
def download_clip(job_id: str, clip_name: str):
    job = manager.jobs.get(job_id)
    if not job or job.status != JobStatus.READY:
        raise HTTPException(404, "Clip not available")
    path = os.path.join(job.clips_dir, clip_name)
    if not os.path.exists(path):
        raise HTTPException(404, "Clip not found")
    return FileResponse(path, media_type="video/mp4", filename=clip_name)


@app.get("/jobs/{job_id}/thumbs/{clip_name}")
def get_thumbnail(job_id: str, clip_name: str):
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    thumb_name = clip_name.replace(".mp4", "_thumb.jpg")
    path = os.path.join(job.clips_dir, thumb_name)
    if not os.path.exists(path):
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")