#!/usr/bin/env bash
set -euo pipefail

# 1) ensure venv activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Please activate venv first:"
  echo "  source venv/Scripts/activate"
  exit 1
fi

# 2) clean old builds
rm -rf build dist *.spec.bak 2>/dev/null || true

# 3) install deps
python -m pip install --upgrade pip
pip install -r requirements-release.txt
pip install pyinstaller

# 4) build
pyinstaller --noconfirm --clean SubtitleOCR_ASS.spec

# 5) create zip
cd dist
rm -f SubtitleOCR_ASS-windows.zip
powershell Compress-Archive SubtitleOCR_ASS SubtitleOCR_ASS-windows.zip
cd ..

echo ""
echo "✅ Done!"
echo "Output folder: dist/SubtitleOCR_ASS/"
echo "Release zip:   dist/SubtitleOCR_ASS-windows.zip"
