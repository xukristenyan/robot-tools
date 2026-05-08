#!/usr/bin/env bash
# Pre-fetch SAM 3 weights from HuggingFace into the local cache, after auth.
# Lets us surface auth/access errors here instead of on first server launch.
#
# Requires `hf auth login` first (and access granted at huggingface.co/facebook/sam3).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[error] $VENV_DIR does not exist. Run \`robot-tools sync\` + \`setup sam3\` first." >&2
    exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

echo "[sam3] Verifying HuggingFace auth and access to facebook/sam3..."
python - <<'PY'
import sys

try:
    from huggingface_hub import whoami, snapshot_download
except ImportError:
    print("[error] huggingface_hub not in venv. Run `robot-tools setup sam3` first.", file=sys.stderr)
    sys.exit(1)

try:
    user = whoami()
except Exception as e:
    print(f"[error] not authenticated. Run `hf auth login` first ({e})", file=sys.stderr)
    sys.exit(1)
print(f"[sam3] HF user: {user.get('name', '?')}")

print("[sam3] Pre-fetching weights from facebook/sam3 (~few GB; cached at ~/.cache/huggingface/)...")
try:
    path = snapshot_download(repo_id="facebook/sam3")
except Exception as e:
    msg = str(e)
    if "gated" in msg.lower() or "403" in msg:
        print("[error] access to facebook/sam3 not granted yet.", file=sys.stderr)
        print("        Request at https://huggingface.co/facebook/sam3 first.", file=sys.stderr)
    raise
print(f"[sam3] cached at {path}")
PY

echo
echo "[done] SAM 3 weights cached locally; first launch will be fast."
