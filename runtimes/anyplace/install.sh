#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"
REAL_PROJECT="$HERE/real"
UPSTREAM_DIR="$HERE/upstream"
COMPAT_PATCH="$HERE/compat/placement-only.patch"
if [[ ! -f "$UPSTREAM_DIR/anyplace/model/transformer/policy.py" ]]; then
    echo "Initialize selected upstream first: robot-tools install <profile>" >&2
    exit 1
fi
if patch --dry-run --forward --silent -d "$UPSTREAM_DIR" -p1 < "$COMPAT_PATCH" >/dev/null 2>&1; then
    patch --forward --batch -d "$UPSTREAM_DIR" -p1 < "$COMPAT_PATCH"
elif patch --dry-run --reverse --silent -d "$UPSTREAM_DIR" -p1 < "$COMPAT_PATCH" >/dev/null 2>&1; then
    echo "AnyPlace compatibility patch already applied."
else
    echo "AnyPlace patch does not match the pinned upstream." >&2
    exit 1
fi
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
uv sync --project "$REAL_PROJECT" --locked
uv pip check --python "$VENV_DIR/bin/python"
"$VENV_DIR/bin/python" "$HERE/doctor.py"
"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"
