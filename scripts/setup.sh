#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  machine-vision - one-shot setup (Linux / macOS)
#  Creates .venv, installs pinned requirements, downloads + exports models.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo
echo "============================================================"
echo " machine-vision setup"
echo " $ROOT"
echo "============================================================"
echo

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "[ERROR] $PYTHON_BIN not found. Install Python 3.11+ first." >&2
    exit 1
}
"$PYTHON_BIN" --version

if [ ! -x ".venv/bin/python" ]; then
    echo "[1/4] creating virtual environment .venv ..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "[1/4] .venv already exists"
fi

PY="$ROOT/.venv/bin/python"

echo "[2/4] upgrading pip ..."
"$PY" -m pip install --upgrade pip --quiet

echo "[3/4] installing requirements (this takes a while, ~2 GB) ..."
"$PY" -m pip install -r requirements.txt

# Mirror of the Windows workaround: easyocr may pull opencv-python-headless.
if ! "$PY" - <<'EOF' >/dev/null 2>&1
import cv2
cv2.namedWindow("t")
cv2.destroyAllWindows()
EOF
then
    echo "[i] restoring the GUI build of OpenCV ..."
    "$PY" -m pip install --force-reinstall --no-deps "opencv-contrib-python==5.0.0.93" --quiet
fi

echo "[4/4] downloading models ..."
"$PY" scripts/download_models.py || {
    echo "[WARN] some models are missing - re-run: $PY scripts/download_models.py" >&2
    exit 1
}

cat <<'MSG'

============================================================
  Done. Launch with:
    source .venv/bin/activate
    python src/realtime_vision.py
============================================================
MSG
