import os
import json
import subprocess
import sys
from typing import List, Tuple


def resource_path(rel_path: str) -> str:
    """
    Works for normal run and PyInstaller.
    For PyInstaller one-dir/one-file, sys._MEIPASS points to temp/app bundle.
    """
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel_path)


def run_cmd(cmd: List[str]) -> str:
    """
    Run command and return stdout (utf-8).
    Auto-use bundled ffmpeg/ffprobe if present next to exe/bundle.
    """
    if cmd and cmd[0] in ("ffmpeg", "ffprobe"):
        # Windows exe name; on mac you would bundle "ffmpeg" without .exe
        candidates = [resource_path(f"{cmd[0]}.exe"), resource_path(cmd[0])]
        local = next((p for p in candidates if os.path.exists(p)), None)
        if local:
            cmd = [local] + cmd[1:]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\nSTDERR:\n{err}")
    return p.stdout.decode("utf-8", errors="replace").strip()


def ffprobe_json(path: str) -> dict:
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path
    ])
    return json.loads(out)


def get_video_info(path: str) -> Tuple[int, int, float]:
    info = ffprobe_json(path)
    vstreams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if not vstreams:
        raise ValueError("No video stream found.")
    vs = vstreams[0]
    width = int(vs["width"])
    height = int(vs["height"])
    duration = float(info["format"].get("duration") or vs.get("duration"))
    return width, height, duration


def extract_preview_frame(video_path: str, out_png: str, t_sec: float):
    """
    Extract ONE frame as a png for ROI preview UI.
    """
    run_cmd([
        "ffmpeg", "-y",
        "-ss", str(max(0.0, t_sec)),
        "-i", video_path,
        "-frames:v", "1",
        out_png
    ])
