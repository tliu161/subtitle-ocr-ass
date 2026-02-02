import os
import re
import json
import shutil
import sys
import types
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import cv2
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein
from opencc import OpenCC

from .video_utils import run_cmd, get_video_info


# -------------------------
# Shims (avoid heavy/unused deps during PaddleOCR import)
# -------------------------
def install_import_shims():
    """
    PaddleOCR import chain touches some training/e2e deps that we don't need for inference.
    Provide tiny stubs so exe doesn't crash if those optional packages are not bundled.
    """
    # imghdr: stdlib, sometimes missed by PyInstaller analysis
    try:
        import imghdr  # noqa: F401
    except ModuleNotFoundError:
        imghdr = types.ModuleType("imghdr")

        def what(file, h=None):
            return None

        imghdr.what = what
        sys.modules["imghdr"] = imghdr

    # skimage stub
    if "skimage" not in sys.modules:
        skimage = types.ModuleType("skimage")
        morphology = types.ModuleType("skimage.morphology")
        _skeletonize = types.ModuleType("skimage.morphology._skeletonize")

        def thin(*args, **kwargs):
            raise RuntimeError(
                "skimage.thin() was called, but skimage is not bundled. "
                "If you need PGNet/E2E, install and bundle scikit-image."
            )

        _skeletonize.thin = thin
        morphology._skeletonize = _skeletonize
        skimage.morphology = morphology

        sys.modules["skimage"] = skimage
        sys.modules["skimage.morphology"] = morphology
        sys.modules["skimage.morphology._skeletonize"] = _skeletonize

    # imgaug stub
    if "imgaug" not in sys.modules:
        imgaug = types.ModuleType("imgaug")
        augmenters = types.ModuleType("imgaug.augmenters")

        class _NotUsed:
            def __getattr__(self, name):
                raise RuntimeError("imgaug augmenter was called, but imgaug is not bundled.")

        # PaddleOCR sometimes uses: import imgaug.augmenters as iaa
        augmenters.iaa = _NotUsed()
        imgaug.augmenters = augmenters

        sys.modules["imgaug"] = imgaug
        sys.modules["imgaug.augmenters"] = augmenters

    # lmdb stub
    if "lmdb" not in sys.modules:
        lmdb = types.ModuleType("lmdb")

        class _LMDBNotUsed:
            def __getattr__(self, name):
                raise RuntimeError("lmdb is not bundled (training-only dependency).")

        lmdb.open = _LMDBNotUsed()
        lmdb.Environment = _LMDBNotUsed()
        sys.modules["lmdb"] = lmdb


# -------------------------
# Data models
# -------------------------
@dataclass
class Item:
    t: float
    text_sc: str
    text_tc: str
    pos: Tuple[int, int]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    pos: Tuple[int, int]


# -------------------------
# Core utils
# -------------------------
def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("丨", "｜")
    return s


def ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:
        cs = 0
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def extract_frames(video: str, frames_dir: str, sample_fps: float, progress_cb=None):
    if progress_cb:
        progress_cb("extract", 0.0, "Extracting frames…")

    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    run_cmd([
        "ffmpeg", "-y",
        "-i", video,
        "-vf", f"fps={sample_fps}",
        os.path.join(frames_dir, "frame_%06d.png")
    ])

    if progress_cb:
        progress_cb("extract", 1.0, "Extracting frames… done")



def roi_crop(img, custom: Tuple[int, int, int, int]):
    h, w = img.shape[:2]
    x, y, cw, ch = custom
    x2 = min(w, x + cw)
    y2 = min(h, y + ch)
    return img[y:y2, x:x2], x, y


def pick_best_text_and_pos(ocr_result, x_off: int, y_off: int) -> Tuple[str, Tuple[int, int]]:
    if ocr_result is None:
        return "", (0, 0)
    if isinstance(ocr_result, list) and len(ocr_result) == 0:
        return "", (0, 0)

    if isinstance(ocr_result, list) and len(ocr_result) == 1 and isinstance(ocr_result[0], list):
        lines = ocr_result[0]
    else:
        lines = ocr_result

    if lines is None:
        return "", (0, 0)

    texts, boxes = [], []
    for line in lines:
        if line is None:
            continue
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue

        box = line[0]
        info = line[1]
        if not info or not isinstance(info, (list, tuple)) or len(info) < 1:
            continue

        text = info[0]
        if not text or not str(text).strip():
            continue

        if not box or not isinstance(box, (list, tuple)) or len(box) < 4:
            continue

        texts.append(clean_text(str(text)))
        boxes.append(box)

    if not texts or not boxes:
        return "", (0, 0)

    xs, ys = [], []
    for box in boxes:
        for p in box:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))

    if not xs or not ys:
        return "", (0, 0)

    merged_text = " ".join(texts).strip()
    cx = int((min(xs) + max(xs)) / 2) + x_off
    cy = int((min(ys) + max(ys)) / 2) + y_off
    return merged_text, (cx, cy)
    
def build_items(
    frames_dir: str,
    sample_fps: float,
    roi_xywh: Tuple[int, int, int, int],
    change_threshold: float,
    opencc: OpenCC,
    debug_first_n: int = 0,
    progress_cb=None,
    progress_every: int = 5,   # 每 5 張回報一次，避免 UI 太頻繁
) -> List[Item]:
    install_import_shims()
    from paddleocr import PaddleOCR  # delayed import

    ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
    total = max(len(frame_files), 1)

    items: List[Item] = []
    prev_text_sc: Optional[str] = None

    if progress_cb:
        progress_cb("ocr", 0.0, f"OCR frames… (0/{total})")

    for i, fn in enumerate(tqdm(frame_files, desc="OCR frames")):
        t = i / sample_fps
        img = cv2.imread(os.path.join(frames_dir, fn))
        if img is None:
            continue

        roi, x_off, y_off = roi_crop(img, roi_xywh)
        res = ocr.ocr(roi, cls=False)

        text_sc, pos = pick_best_text_and_pos(res, x_off, y_off)
        text_sc = text_sc.strip()

        if debug_first_n and i < debug_first_n:
            print(f"[t={t:.2f}s] OCR='{text_sc}' pos={pos}")

        if not text_sc:
            prev_text_sc = text_sc
        else:
            text_tc = opencc.convert(text_sc)

            if prev_text_sc:
                maxlen = max(len(prev_text_sc), len(text_sc), 1)
                dist = Levenshtein.distance(prev_text_sc, text_sc) / maxlen
                if dist < change_threshold:
                    items.append(Item(t=t, text_sc=prev_text_sc, text_tc=opencc.convert(prev_text_sc), pos=pos))
                else:
                    items.append(Item(t=t, text_sc=text_sc, text_tc=text_tc, pos=pos))
                    prev_text_sc = text_sc
            else:
                items.append(Item(t=t, text_sc=text_sc, text_tc=text_tc, pos=pos))
                prev_text_sc = text_sc

        # ---- progress update ----
        if progress_cb and (i % progress_every == 0 or i == total - 1):
            progress_cb("ocr", (i + 1) / total, f"OCR frames… ({i+1}/{total})")

    return items



def items_to_segments(items: List[Item], sample_fps: float, hold_gap: float) -> List[Segment]:
    if not items:
        return []

    segments: List[Segment] = []
    cur_text = items[0].text_tc
    cur_start = items[0].t
    pos_list = [items[0].pos]
    last_t = items[0].t

    def median_pos(ps):
        xs = sorted(p[0] for p in ps)
        ys = sorted(p[1] for p in ps)
        return (xs[len(xs) // 2], ys[len(ys) // 2])

    for it in items[1:]:
        same = (it.text_tc == cur_text)
        close_in_time = (it.t - last_t) <= (1.5 / sample_fps)

        if same and close_in_time:
            pos_list.append(it.pos)
            last_t = it.t
            continue

        segments.append(Segment(start=cur_start, end=last_t + hold_gap, text=cur_text, pos=median_pos(pos_list)))

        cur_text = it.text_tc
        cur_start = it.t
        pos_list = [it.pos]
        last_t = it.t

    segments.append(Segment(start=cur_start, end=last_t + hold_gap, text=cur_text, pos=median_pos(pos_list)))
    return segments


def fill_gaps_upto(segments: List[Segment], max_gap: float) -> List[Segment]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start)
    for i in range(len(segments) - 1):
        gap = segments[i + 1].start - segments[i].end
        if gap < 0:
            segments[i].end = segments[i + 1].start
        elif 0 < gap <= max_gap:
            segments[i].end = segments[i + 1].start
    return segments


def write_ass(segments: List[Segment], out_path: str, play_res_x: int, play_res_y: int):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft JhengHei,44,&H00FFFFFF,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def esc(s: str) -> str:
        s = s.replace("\\", "\\\\")
        s = s.replace("{", "\\{").replace("}", "\\}")
        s = s.replace("\n", "\\N")
        return s

    lines = [header]
    for seg in segments:
        x, y = seg.pos
        x = max(0, min(play_res_x, x))
        y = max(0, min(play_res_y, y))
        txt = esc(seg.text.strip())
        if not txt:
            continue
        override = f"{{\\pos({x},{y})}}"
        lines.append(f"Dialogue: 0,{ass_time(seg.start)},{ass_time(seg.end)},Default,,0,0,0,,{override}{txt}\n")

    with open(out_path, "w", encoding="utf-8-sig") as f:
        f.write("".join(lines))

def process_video_to_ass(
    video_path: str,
    roi_xywh: Tuple[int, int, int, int],
    out_ass: Optional[str] = None,
    sample_fps: float = 3.0,
    change_threshold: float = 0.18,
    hold_gap: float = 0.25,
    fill_gaps: float = 2.0,
    keep_frames: bool = False,
    debug_first_n: int = 0,
    progress_cb=None,
) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    base, _ = os.path.splitext(video_path)
    out_ass = out_ass or f"{base}_tc.ass"
    frames_dir = "_frames_tmp"

    w, h, dur = get_video_info(video_path)
    print(f"Video: {video_path}  {w}x{h}  {dur:.1f}s")
    print(f"ROI: {roi_xywh}  |  fps={sample_fps}  thr={change_threshold}  hold={hold_gap}  fill={fill_gaps}")

    # ---- Stage 1: extract frames (0–10%) ----
    if progress_cb:
        progress_cb("stage", 0.0, "Stage 1/3: Extracting frames…")
        progress_cb("overall", 0.0, "Stage 1/3: Extracting frames…")

    def stage_extract_cb(_phase, frac, msg):
        # map 0..1 -> 0..0.10
        if progress_cb:
            progress_cb("overall", 0.10 * frac, msg)

    extract_frames(video_path, frames_dir, sample_fps, progress_cb=stage_extract_cb)

    # ---- Stage 2: OCR (10–95%) ----
    if progress_cb:
        progress_cb("stage", 0.0, "Stage 2/3: OCR…")
        progress_cb("overall", 0.10, "Stage 2/3: OCR…")

    cc = OpenCC("s2t")

    def stage_ocr_cb(_phase, frac, msg):
        # map 0..1 -> 0.10..0.95
        if progress_cb:
            progress_cb("overall", 0.10 + 0.85 * frac, msg)

    items = build_items(
        frames_dir=frames_dir,
        sample_fps=sample_fps,
        roi_xywh=roi_xywh,
        change_threshold=change_threshold,
        opencc=cc,
        debug_first_n=debug_first_n,
        progress_cb=stage_ocr_cb,
        progress_every=5,
    )

    segments = items_to_segments(items, sample_fps=sample_fps, hold_gap=hold_gap)
    if fill_gaps and fill_gaps > 0:
        segments = fill_gaps_upto(segments, max_gap=fill_gaps)

    # ---- Stage 3: write file (95–100%) ----
    if progress_cb:
        progress_cb("stage", 0.0, "Stage 3/3: Writing ASS…")
        progress_cb("overall", 0.95, "Stage 3/3: Writing ASS…")

    write_ass(segments, out_ass, play_res_x=w, play_res_y=h)

    if progress_cb:
        progress_cb("overall", 1.0, "Done ✅")

    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return out_ass
