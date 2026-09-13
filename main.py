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
from pydantic import BaseModel

from job_manager import JobManager, JobStatus

app = FastAPI(title="Streamer Clip Generator")
manager = JobManager(jobs_root=os.path.join(os.path.dirname(__file__), "jobs"))


class LiveRequest(BaseModel):
    url: str


def build_record_command(url: str, out_path: str):
    """
    PRODUCTION command: pull a live  stream with yt-dlp until it ends
    naturally or is terminated. --impersonate chrome avoids some of the
    bot-detection issues 's downloads have had recently.
    """
    return ["yt-dlp", url, "-f", "best", "-o", out_path, "--impersonate", "chrome", "--no-part"]


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
