# Streamer Clip Generator — Web App

A minimal backend + frontend that lets a streamer paste a live URL (or upload
a finished VOD), automatically records/stops, and generates downloadable
vertical short clips — reusing `auto_clip.py` and `vertical_reframe.py`.

## Setup
```bash
pip install -r requirements.txt
# Also required, as before: ffmpeg installed, yt-dlp installed
```

## Run it
```bash
uvicorn main:app --reload --port 8000
```
Then open http://localhost:8000 in a browser.

## How it works

1. **Paste a live URL, or upload a VOD.** Both paths converge on the same
   pipeline once a video file exists — the app doesn't care which source it
   came from.
2. **Live recording** runs yt-dlp in the background. It stops in one of two
   ways, both handled identically:
   - You click "Stop Stream" in the UI.
   - The stream ends on its own and yt-dlp exits by itself.
   A background watcher thread detects either case the same way (it just
   waits for the recording process to exit, for any reason) and immediately
   moves the job into "processing."
3. **Processing** runs the exact same `auto_clip.py` highlight-detection +
   clip-cutting logic, then `vertical_reframe.py` on every clip, in the
   background — the streamer sees a live status update while this happens.
4. **The raw recording is deleted automatically** the moment clips are
   generated (success or failure) — it's never kept around. There's also a
   background safety-net sweeper that force-deletes any raw video older than
   6 hours, in case a job ever gets stuck, so storage never silently grows.
5. **Clips are downloadable** individually from the finished job.

## Storage notes (read this before deploying for real users)

This version stores everything on local disk under `jobs/{job_id}/`, which
is fine for a single-server MVP with a handful of concurrent streamers. If
you scale to many streamers recording at once, the two things to change are:

- **Move storage to object storage** (Cloudflare R2 or Backblaze B2 both
  have generous free/cheap tiers) instead of local disk, so you're not
  bottlenecked by one server's disk space.
- **Add a lifecycle rule on the bucket** that force-deletes anything older
  than ~24 hours, as a second safety net beyond the in-app sweeper — belt
  and suspenders, since storage costs (and privacy risk) compound silently
  if a bug ever stops cleanup from running.

## Other improvements worth adding next
- **Chunked recording**: record in fixed segments (e.g. every 10 minutes)
  instead of one giant growing file — protects against losing a whole 3-hour
  recording if the process crashes, and lets you start finding highlights in
  earlier segments *before* the stream even ends.
- **Auth**: right now anyone can submit any URL. Add a login + verify the
  streamer owns the channel before recording it.
- **Job queue** (Celery/RQ + Redis) instead of raw threads once you have
  more concurrent jobs than one server's CPU can encode at once.
- **Notifications**: email/webhook when clips are ready, since a 2-hour
  stream means a long wait — don't make the streamer babysit a browser tab.
