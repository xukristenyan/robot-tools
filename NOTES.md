# Adding a new server to robot-tools

Step-by-step recipe to add a new model server (e.g. SAM, OwlViT, MoLMo).
Keep this in sync with the actual code; treat existing `graspgen/` and
`fastfs/` as the reference implementations.

## 0. Architecture invariants (don't fight them)

- **Multi-venv**: each `servers/<name>/` is its own uv project with its own
  `.venv`. Heavy ML deps are scoped to that venv only. Different servers can
  use different Python versions, different torch versions — they don't
  collide.
- **Light client SDK**: `src/robot_tools/` (the pip-installable package)
  only ever has client-side deps. Never put torch / model libs there.
- **Contract is shared**: client and server both import the same pydantic
  schema from `src/robot_tools/contracts/<name>.py`. One source of truth.
- **Server returns raw model output**: keep the server agnostic to
  per-camera / per-environment knobs. Wrap conversions on the client side
  (see `clients/fastfs.py`'s `disparity_to_depth` helper).
- **Per-call params via client kwargs**, not a server-side config file.
  Server-startup args (gripper, model id) go in `services.yaml`'s
  `extra_args`.

## 1. Files to create

For server `<X>`:

```
src/robot_tools/contracts/<X>.py      schema (pydantic)
src/robot_tools/clients/<X>.py        client class (inherits BaseClient)
src/robot_tools/registry.py           +1 entry  (edit existing file)

servers/<X>/                          uv project (independent venv)
├── pyproject.toml                    light deps + Python pin
├── .python-version                   pin Python version
├── .gitignore                        ignore weights/checkpoints dir
├── README.md                         bring-up + status
├── server.py                         BaseServer subclass with @handler-decorated methods
├── pipeline.py                       headless wrapper around the upstream model
├── setup.sh                          manual install steps (torch + ML deps)
├── download.sh                       fetch model weights
└── upstream/                         git submodule of upstream model repo
```

## 2. Step-by-step

### Step 1 — submodule upstream

```bash
cd ~/CodeLib/robot-tools
git submodule add <upstream-git-url> servers/<X>/upstream
```

### Step 2 — write the contract (`contracts/<X>.py`)

```python
from typing import Literal
import numpy as np
from robot_tools.contracts.base import BaseRequest, BaseResponse


class FooRequest(BaseRequest):
    action: Literal["foo"] = "foo"          # MUST be a Literal default
    image: np.ndarray                        # numpy fields OK; allowed via base config
    threshold: float = 0.5                   # per-call tunables with sensible defaults


class FooResponse(BaseResponse):
    masks: np.ndarray
    scores: np.ndarray
```

**Notes:**
- Inherit from `BaseRequest` / `BaseResponse` (not `BaseModel`).
- `action` MUST be `Literal["..."] = "..."` — `BaseServer` reads its default
  to build the routing table. No default = handler registration fails.
- Numpy fields are allowed because the base classes set
  `arbitrary_types_allowed=True`.

### Step 3 — write the client (`clients/<X>.py`)

```python
import numpy as np
from robot_tools.clients.base import BaseClient
from robot_tools.contracts.<X> import FooRequest, FooResponse


class FooClient(BaseClient):
    def foo(self, image: np.ndarray, *, threshold: float = 0.5) -> FooResponse:
        req = FooRequest(image=np.asarray(image), threshold=threshold)
        return self._request(req, FooResponse)
```

**Notes:**
- One client method per request type.
- Each method returns the full pydantic response (not a tuple) so callers
  get IDE-friendly access to all fields.
- Add static helpers (e.g. `disparity_to_depth`) at module-level when the
  client commonly post-processes outputs.

### Step 4 — write the server project

#### `servers/<X>/pyproject.toml`

```toml
[project]
name = "<X>-server"
version = "0.1.0"
description = "<X> ZMQ server."
readme = "README.md"
requires-python = ">=3.10,<3.11"   # whatever the upstream needs

dependencies = [
    "robot-tools",
]

[tool.uv.sources]
robot-tools = { path = "../..", editable = true }

[build-system]
requires = ["uv_build>=0.10.12,<0.11.0"]
build-backend = "uv_build"

[tool.uv]
package = false
# If you hit numpy 1.x / 2.x compat bugs (e.g. with old torch), pin here:
# override-dependencies = ["numpy==1.26.4"]
```

#### `servers/<X>/.python-version`

Just the version: `3.10` (or whatever).

#### `servers/<X>/.gitignore`

```
weights/         # or whatever your download.sh creates
ModelDir/        # match what's in download.sh
```

#### `servers/<X>/server.py`

```python
import argparse, logging
import numpy as np

from robot_tools.contracts.<X> import FooRequest, FooResponse
from robot_tools.server_base import BaseServer, handler

logger = logging.getLogger(__name__)


class FooServer(BaseServer):
    def __init__(self, host="127.0.0.1", port=5555, *, mock=False, model_id="default"):
        super().__init__(host=host, port=port)
        self._mock = mock
        if mock:
            logger.warning("MOCK mode")
            self.pipeline = None
        else:
            from pipeline import FooPipeline
            self.pipeline = FooPipeline(model_id)

    @handler(FooRequest)
    def foo(self, req):
        if self._mock:
            return FooResponse(masks=np.zeros((1,)), scores=np.zeros((1,)))
        result = self.pipeline.run(req.image, threshold=req.threshold)
        return FooResponse(masks=result["masks"], scores=result["scores"])


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model", default="default")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    FooServer(host=args.host, port=args.port, mock=args.mock,
              model_id=args.model).serve_forever()


if __name__ == "__main__":
    main()
```

**Notes:**
- Always provide `--mock` so users can test wiring without installing real
  deps.
- Heavy `from pipeline import ...` import goes inside `__init__` (or
  guarded by `if not mock`), so mock mode runs even when ML deps are
  missing.
- CLI args become per-server-startup config; per-call config goes in the
  request schema.

#### `servers/<X>/pipeline.py`

The headless wrapper around the upstream model. Two patterns depending
on how upstream is packaged:

**Pattern A — upstream HAS setup.py / pyproject.toml** (graspgen):
```python
from grasp_gen.X import Y    # imports work after `setup.sh` ran uv pip install -e ./upstream
```

**Pattern B — upstream is just a research repo** (fastfs):
```python
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "upstream"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from Utils import ...        # import directly from the cloned tree
```

In both patterns, weights live at `<HERE>/weights/` (or similar) by
default; allow override via env var:

```python
def _weights_dir() -> Path:
    raw = os.getenv("FOO_WEIGHTS_DIR")
    candidate = Path(raw).expanduser().resolve() if raw else _HERE / "weights"
    if not candidate.exists():
        raise RuntimeError(f"weights not found at {candidate}; run download.sh")
    return candidate
```

#### `servers/<X>/setup.sh`

Script that runs the manual install steps `uv sync` can't express. Two
patterns:

**Pattern A — upstream is pip-installable** (graspgen):
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"
[[ -d "$VENV_DIR" ]] || { echo "no .venv; run robot-tools sync first" >&2; exit 1; }
export VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$PATH"

# Run from upstream/ so uv reads upstream's [tool.uv] (e.g. find-links).
( cd "$HERE/upstream" && uv pip install -e . )

# Pin numpy explicitly if torch's numpy ABI matters:
# uv pip install --reinstall-package numpy 'numpy==1.26.4'
```

**Pattern B — upstream needs torch + requirements.txt** (fastfs):
```bash
uv pip install torch==2.6.0 torchvision==0.21.0 xformers \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install -r "$HERE/upstream/requirements.txt"
```

Always use `set -euo pipefail` + check `.venv` exists at the top.

#### `servers/<X>/download.sh`

Script that fetches model weights. Pattern depends on host:

**HuggingFace (git-lfs)**:
```bash
git clone https://huggingface.co/<user>/<repo> "$DEST"
```

**Google Drive folder** (needs `gdown` from setup.sh):
```bash
gdown --folder "$GDRIVE_URL" -O "$DEST"
```

Make idempotent: `if [[ -d "$DEST/.git" ]]; then exit 0; fi` for HF.

### Step 5 — add to registry (`src/robot_tools/registry.py`)

```python
"foo": ServerSpec(
    name="foo",
    project_dir=REPO_ROOT / "servers" / "foo",
    entry=["python", "server.py"],
    default_port=5555,                # pick an unused port
    gpu_required=True,
),
```

Pick a default port that doesn't collide with other servers in `services.yaml`.

### Step 6 — make scripts executable

```bash
chmod +x servers/<X>/setup.sh servers/<X>/download.sh
```

## 3. Verify (from a downstream project)

```bash
cd ~/CodeLib/projects/<some-test-project>

# Edit services.yaml to add your server with --mock initially:
#   - name: foo
#     port: 5555
#     extra_args: ["--mock"]

uv run robot-tools list                  # confirm registry sees it
uv run robot-tools sync   services.yaml  # creates servers/<X>/.venv
uv run robot-tools launch services.yaml  # mock server boots
uv run python -c "from robot_tools.clients.foo import FooClient; \
                  c=FooClient(host='127.0.0.1', port=5555); \
                  print(c.foo(...))"

# Then for real install:
uv run robot-tools setup    foo
uv run robot-tools download foo
# drop --mock from services.yaml
uv run robot-tools launch services.yaml
```

## 4. Pitfalls I've already hit (don't repeat)

### `uv sync` strips `uv pip install`-ed packages
After `setup.sh` does `uv pip install -e ./upstream`, a later `uv sync`
will remove `grasp_gen` etc. because they're not in `pyproject.toml`.
**Already fixed in `launcher.py`**: `_sync_one` uses `uv sync --inexact`.
If you call `uv sync` manually, also use `--inexact`.

### `uv run` strips them too
Same problem with `uv run` re-syncing. **Already fixed in launcher**:
`_spawn` uses `uv run --no-sync`. If you launch the server by hand, use
`uv run --no-sync python server.py` after `setup.sh`.

### Numpy 2.x / torch 1.x compat
Some torch builds (e.g. 2.1.0 from graspgen) were compiled against
numpy 1.x and warn loudly when used with numpy 2.x. Fix by adding
`override-dependencies = ["numpy==1.26.4"]` to that server's
`[tool.uv]`. Don't put this in the root SDK — it's per-server.

### Editable installs break on directory rename
If you `mv` the `robot-tools/` folder, the `uv pip install -e ./upstream`
metadata still points at the old path. After any rename, re-run
`robot-tools setup <name>` for each server, and `uv sync --inexact` to
fix the `robot-tools` editable too.

### Putting heavy deps in `pyproject.toml` `[real]` extra fails
Tempting: `[project.optional-dependencies] real = ["grasp_gen"]`. But
uv's lockfile resolver tries to BUILD all extras even when not selected
with `--extra`, which triggers torch-cluster's missing-build-dep error.
**Use `setup.sh` (imperative `uv pip install`) instead** — that's what
upstream READMEs do.

### `find-links` lives in the upstream's pyproject, not ours
GraspGen (and likely others) ship `[tool.uv] find-links = ["..."]` in
their pyproject pointing at PyG / cu124 wheel indices. `setup.sh` must
`cd upstream/` before `uv pip install -e .` so uv reads that config.
Otherwise build deps for torch-cluster etc. fail.

### Server-side rendering is the wrong layer
Don't put viser / matplotlib / pyglet helpers in the server's
`pipeline.py`. Server is headless. Move visualization to client (or a
separate utility module).

### Upstream `dependencies` is incomplete → ModuleNotFoundError at runtime
Research repos often split deps across `[dev]`/`[notebooks]`/`[train]`
extras and leave `[project] dependencies` skeletal. Result: `uv pip install
-e .` succeeds but the server crashes on first import (`einops`,
`hydra-core`, `pycocotools`, `scipy`, `scikit-image`, `psutil`, `opencv`,
etc. typically missing). SAM 3 was the offender for us, but FFS / SAM 2 /
others have the same shape.

Fix: in `setup.sh` after `uv pip install -e ./upstream`, **explicitly
install the runtime subset**:
```bash
uv pip install \
    einops hydra-core pycocotools \
    scipy scikit-image scikit-learn \
    psutil opencv-python-headless
```
Use `opencv-python-headless` (not `opencv-python`) since servers don't
need X11/Qt. To find the actual list for a new server, grep `import` in
`upstream/<pkg>/` and diff against the venv after a bare `pip install -e .`.

### System Python lacks dev headers → Triton / JIT fails
If your server uses anything that JIT-compiles C extensions at runtime
(Triton, custom CUDA kernels, some torch features), and uv picks the
system `/usr/bin/python3.X` for the venv, you'll hit:
```
fatal error: Python.h: No such file or directory
```
Reason: most Linux distros ship Python without `python3.X-dev`, so the
system `/usr/include/python3.X/` is empty. uv's *managed* Python
downloads (from `uv python install 3.X`) bundle full dev headers.

Fix in the server's `pyproject.toml`:
```toml
[tool.uv]
python-preference = "only-managed"
```
This forces uv to use its own downloaded interpreter (run
`uv python install 3.X` first if missing). Then `rm -rf .venv && uv sync
--inexact` and re-run `setup.sh`. **Apply this to any server using
torch ≥ 2.4 or Triton-backed ops** (fastfs is the example).

## 5. When you're done

The whole flow above takes ~30 minutes per new server, **without
touching any existing code** — just new files plus 1 line in
`registry.py`. If you find yourself modifying `BaseServer`,
`BaseClient`, or `launcher.py`, stop and ask whether the abstraction
is wrong.
