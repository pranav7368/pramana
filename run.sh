#!/bin/sh
# Works from any current directory, including paths containing spaces.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$task_root"
if [ -x "$task_root/.venv/bin/python" ]; then
    exec "$task_root/.venv/bin/python" "$task_root/run_demo.py" "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "$task_root/run_demo.py" "$@"
else
    printf '%s\n' 'Install Python 3.12, then run: sh run.sh'
    exit 1
fi
