#!/usr/bin/env bash
# Delete the conversation database (shared log, memories, moods, relationships).
#   ./reset.sh                    delete the DB set in config.toml ([store] path)
#   ./reset.sh --all              delete every *.db in data/ (old DBs too)
#   ./reset.sh --config FILE      read [store] path from FILE instead of config.toml
#   ./reset.sh -y                 do not ask for confirmation
# Close the app before running this.
set -euo pipefail
cd "$(dirname "$0")"

ALL=0; YES=0; CONFIG="config.toml"
while [ $# -gt 0 ]; do
    case "$1" in
        --all) ALL=1 ;;
        -y|--yes) YES=1 ;;
        --config) shift; CONFIG="${1:?--config needs a file}" ;;
        -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python; fi

targets=()
if [ "$ALL" -eq 1 ]; then
    shopt -s nullglob
    targets=(data/*.db)
else
    if ! db="$("$PY" -c 'import sys, tomllib
with open(sys.argv[1], "rb") as f:
    print(tomllib.load(f).get("store", {}).get("path", "data/imaginary_friend.db"))' "$CONFIG")" || [ -z "$db" ]; then
        echo "could not read [store] path from $CONFIG (needs Python 3.11+)" >&2
        exit 1
    fi
    db="${db%$'\r'}"
    [ -e "$db" ] && targets=("$db")
fi

if [ ${#targets[@]} -eq 0 ]; then
    echo "nothing to delete."
    exit 0
fi

echo "will delete:"
for db in "${targets[@]}"; do
    for f in "$db" "$db-wal" "$db-shm" "$db-journal"; do [ -e "$f" ] && echo "  $f"; done
done

if [ "$YES" -ne 1 ]; then
    read -r -p "really delete? [y/N] " ans
    case "$ans" in y|Y|yes|YES) ;; *) echo "cancelled."; exit 1 ;; esac
fi

for db in "${targets[@]}"; do
    rm -f -- "$db" "$db-wal" "$db-shm" "$db-journal"
done
echo "done. a fresh DB will be created on next start."
