FROM python:3.11-slim

# ffmpeg is required by auto_clip.py and vertical_reframe.py.
# git is needed because yt-dlp updates frequently and pip's PyPI release
# can lag behind fixes for sites like Kick — installing from GitHub main
# gets you the latest fixes (e.g. the Kick VOD-URL bug from before).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade "yt-dlp[default]@git+https://github.com/yt-dlp/yt-dlp.git"

COPY . .

# Render sets $PORT at runtime — don't hardcode 8000.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]