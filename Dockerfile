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
#
# unzip is needed to extract the Deno binary below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp (2025.11.12+) requires an external JavaScript runtime to solve
# YouTube's signature/challenge code — without one you get the
# "No supported JavaScript runtime could be found" warning and YouTube
# downloads can fail or be missing formats. This does NOT affect Kick,
# but if this app is ever used for YouTube live/VOD too, it's required.
# Deno is yt-dlp's recommended/default runtime. Installed as a plain
# binary (not via deno's own install script) so the exact version is
# pinned and reproducible across builds.
RUN curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip \
    && unzip -o /tmp/deno.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/deno \
    && rm /tmp/deno.zip \
    && deno --version

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade "yt-dlp[default]@git+https://github.com/yt-dlp/yt-dlp.git"
# The yt-dlp-ejs package is the piece that actually invokes the JS runtime
# (Deno, above) to solve YouTube's challenges. Both are required together —
# installing only one still produces the warning.
RUN pip install --no-cache-dir --upgrade yt-dlp-ejs

# Verify the JS runtime is actually wired up correctly at BUILD time, not
# discovered later when a streamer is waiting on a YouTube pull. This will
# fail the build (not just warn) if Deno isn't detected.
RUN yt-dlp --js-runtimes deno --verbose --simulate "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>&1 \
    | grep -q "JS runtimes: deno" \
    && echo "Deno JS runtime OK" \
    || (echo "ERROR: Deno JS runtime not detected by yt-dlp" && exit 1)

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