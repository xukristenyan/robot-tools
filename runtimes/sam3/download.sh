#!/usr/bin/env bash
# Pre-fetch SAM 3 weights from HuggingFace into the local cache, after auth.
# Lets us surface auth/access errors here instead of on first server launch.
#
# Requires `hf auth login` first (and access granted at huggingface.co/facebook/sam3).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[error] $VENV_DIR does not exist. Run \`robot-tools install\` first." >&2
    exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

echo "[sam3] Verifying HuggingFace auth and access to facebook/sam3..."
python - <<'PY'
import sys

try:
    from huggingface_hub import hf_hub_download, whoami
except ImportError:
    print("[error] huggingface_hub not in venv. Run `robot-tools install` first.", file=sys.stderr)
    sys.exit(1)

try:
    user = whoami()
except Exception as e:
    print(f"[error] not authenticated. Run `hf auth login` first ({e})", file=sys.stderr)
    sys.exit(1)
print(f"[sam3] HF user: {user.get('name', '?')}")

print("[sam3] Pre-fetching sam3.pt from facebook/sam3 (cached by HuggingFace Hub)...")
try:
    hf_hub_download(repo_id="facebook/sam3", filename="config.json")
    path = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
except Exception as e:
    msg = str(e)
    if "gated" in msg.lower() or "403" in msg:
        print("[error] access to facebook/sam3 not granted yet.", file=sys.stderr)
        print("        Request at https://huggingface.co/facebook/sam3 first.", file=sys.stderr)
    raise
print(f"[sam3] checkpoint cached at {path}")
PY

echo
echo "[done] SAM 3 weights cached locally; first launch will be fast."
