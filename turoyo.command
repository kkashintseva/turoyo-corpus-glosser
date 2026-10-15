#!/bin/bash
# Turoyo Corpus Glosser — one-step macOS launcher.
cd "$(dirname "$0")" || exit 1
PY=".venv/bin/turoyo-glosser"
if [ ! -x "$PY" ]; then
  echo "Not installed yet. Run once:"
  echo "  cd \"$(pwd)\""
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -e ."
  echo
  read -r -p "Press Enter to close..."
  exit 1
fi
export GLOSSER_PROFILE="${GLOSSER_PROFILE:-flex}"
echo "Turoyo Corpus Glosser: http://127.0.0.1:8000"
echo "Stop with Ctrl+C."
exec "$PY" edit --host 127.0.0.1 --port 8000
