#!/usr/bin/env bash
# Start llm-imaginary-friend. Extra args pass through (e.g. ./go.sh --no-idle).
#   ./go.sh             Japanese edition (config.toml, VOICEVOX)
#   ./go.sh en          English edition (config.en.toml, Kokoro TTS on the CPU)
#   ./go.sh en --cli    remaining args pass through to imaginary_friend.main
set -euo pipefail
cd "$(dirname "$0")"

LANG_ARGS=()
NEED_EN=0
case "${1:-}" in
    en) LANG_ARGS=(--config config.en.toml); NEED_EN=1; shift ;;
    ja) shift ;;
esac

# .venv layout differs: Windows (Git Bash) uses Scripts/, Linux/macOS use bin/
if [ -x ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    echo "[first-time setup] creating .venv ..."
    # Pick a Python that actually runs and is 3.11+. "command -v" is not enough:
    # on Windows, python3 is often the Microsoft Store stub, which is found but fails to run.
    BASE_PY=()
    for cand in "python3" "python" "py -3"; do
        read -r -a c <<< "$cand"
        if "${c[@]}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
            BASE_PY=("${c[@]}"); break
        fi
    done
    if [ ${#BASE_PY[@]} -eq 0 ]; then
        echo "[first-time setup] Python 3.11 or newer was not found. Please install it and try again." >&2
        exit 1
    fi
    # if anything below fails, remove the half-made .venv so the next run starts over
    # (otherwise the next run finds .venv/.../python and skips the install)
    fail_setup() { echo "[first-time setup] failed; removing .venv" >&2; rm -rf .venv; exit 1; }
    "${BASE_PY[@]}" -m venv .venv || fail_setup
    if [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"; else PY=".venv/bin/python"; fi
    "$PY" -m pip install -q -r requirements.txt || fail_setup
fi

if [ "$NEED_EN" -eq 1 ] && ! "$PY" -c "import onnxruntime, misaki, g2p_en, en_core_web_sm" >/dev/null 2>&1; then
    echo "[first-time setup] installing the English edition packages (requirements-en.txt) ..."
    # a .venv made by uv has no pip (spacy download also needs pip), so add it first
    "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade
    "$PY" -m pip install -q -r requirements-en.txt
    "$PY" -m spacy download en_core_web_sm
fi

# ${arr[@]+...}: an empty array under set -u is an error in bash 3.2 (macOS)
exec "$PY" -m imaginary_friend.main ${LANG_ARGS[@]+"${LANG_ARGS[@]}"} "$@"
