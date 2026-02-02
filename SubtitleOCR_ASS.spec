# SubtitleOCR_ASS.spec
# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

PROJECT_ROOT = os.path.abspath(os.getcwd())

datas = []
binaries = []
hiddenimports = []
extra_binaries = []

# ---- Bundle ffmpeg/ffprobe next to the exe (recommended) ----
# Put ffmpeg.exe + ffprobe.exe under ./bin/
BIN_DIR = os.path.join(PROJECT_ROOT, "bin")
for fn in ("ffmpeg.exe", "ffprobe.exe"):
    p = os.path.join(BIN_DIR, fn)
    if os.path.exists(p):
        extra_binaries.append((p, "."))  # copy to dist root

# ---- Collect heavy deps ----
# collect_all returns: datas, binaries, hiddenimports
for pkg in ("paddle", "paddleocr", "Cython"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# ---- Extra hidden imports often missed ----
hiddenimports += [
    "pyclipper",
    "imgaug",
    "imgaug.augmenters",
    "skimage",
    "skimage.morphology",
    "skimage.morphology._skeletonize",
    "lmdb",
    "imghdr",
    # PaddleOCR sometimes imports table metric deps during collect step
    "apted",
]

# ---- Include our app package ----
datas += [(os.path.join(PROJECT_ROOT, "app"), "app")]

# ---- Include extra binaries ----
binaries += extra_binaries

# ---- De-dup hiddenimports ----
hiddenimports = list(dict.fromkeys(hiddenimports))

a = Analysis(
    ["app/main.py"],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SubtitleOCR_ASS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed
    disable_windowed_traceback=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="SubtitleOCR_ASS",
)
