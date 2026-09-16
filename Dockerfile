FROM python:3.11-slim

# ffmpeg is required by auto_clip.py and vertical_reframe.py.
# git is needed because yt-dlp updates frequently and pip's PyPI release
# can lag behind fixes for sites like Kick — installing from GitHub main
# gets you the latest fixes (e.g. the Kick VOD-URL bug from before).
#
# libgl1 / libglib2.0-0 are required by OpenCV. Even the "headless" build
# links against these shared libraries, and python:3.11-slim does NOT
# include them — this is a classic works-locally-fails-in-Docker cause,
# since desktop Linux/Windows/macOS already have them installed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade "yt-dlp[default]@git+https://github.com/yt-dlp/yt-dlp.git"

# Fail the BUILD (not a user's job at runtime) if OpenCV isn't fully working.
# Without this check, a broken cv2 only surfaces when a streamer is waiting
# on their clips — much better to catch it here.
RUN python -c "import cv2; assert hasattr(cv2, 'CascadeClassifier'), 'cv2 install is broken'; \
    import os; p = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'; \
    assert os.path.exists(p), 'haar cascade data file missing'; \
    print('OpenCV OK:', cv2.__version__)"

COPY . .

# Render sets $PORT at runtime — don't hardcode 8000.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]