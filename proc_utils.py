"""
proc_utils.py — run heavy subprocesses without starving the web server.

WHY THIS EXISTS:
  Clip processing (ffmpeg encode, OpenCV decode) runs in a background
  thread inside the SAME container as uvicorn. On a fractional-CPU host
  (e.g. 0.1 CPU on a free tier), ffmpeg will happily consume the entire
  CPU quota. uvicorn then can't respond to the platform's health checks,
  the platform declares the service unhealthy, restarts the container, and
  the user sees a 502 — with any in-flight job lost.

  Running ffmpeg at a low scheduling priority (nice) means the kernel
  preempts it whenever uvicorn has work to do. ffmpeg still gets all the
  idle CPU, so throughput barely changes, but health checks keep getting
  answered.
"""

import os
import subprocess
import sys

# How much to deprioritize heavy work. 10 is a solid default: clearly
# below the web server, still well above idle.
NICE_LEVEL = int(os.environ.get("CLIP_NICE", "10"))


def _demote():
    """Runs in the child process after fork, before exec."""
    try:
        os.nice(NICE_LEVEL)
    except Exception:
        pass  # never let a priority tweak break the actual work
    try:
        # Detach into its own process group so the whole tree can be
        # signalled together when a job is cancelled.
        os.setsid()
    except Exception:
        pass


def popen_kwargs(extra=None):
    """Build Popen kwargs that deprioritize the child where supported."""
    kwargs = dict(extra or {})
    if sys.platform == "win32":
        kwargs.setdefault(
            "creationflags",
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS,
        )
    else:
        kwargs.setdefault("preexec_fn", _demote)
    return kwargs


def run(cmd, **kwargs):
    """Drop-in replacement for subprocess.run that runs at low priority."""
    return subprocess.run(cmd, **popen_kwargs(kwargs))