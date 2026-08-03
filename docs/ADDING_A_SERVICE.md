# Adding a service

This is the maintenance checklist for introducing a new logical service and
its independently deployable runtime. Read the
[architecture invariants](ARCHITECTURE.md) first, then use the simplest
existing service with similar inputs as the reference implementation.

> [!NOTE]
> Add a new service when the capability needs its own identity, compatibility
> boundary, or runtime. If the change extends an existing compatible API, add
> an action to that service instead.

## Decide the public boundary

Write these decisions down before creating files:

| Decision | Example |
|---|---|
| Stable service ID | `foo` |
| Public actions | `infer`, `metadata` |
| Request and response shapes | Image in, masks and scores out |
| Per-call options | Threshold, maximum result count |
| Startup options | Model ID, default gripper |
| Runtime requirements | Python version, Torch/CUDA, system libraries |
| External artifacts | Upstream repository, checkpoints, credentials |
| Default port | An unused registry port |
| Fake behavior | Deterministic, contract-valid output |

Keep product-specific camera objects, visualization, and orchestration outside
the service API. Contracts should use portable values such as NumPy arrays,
numbers, strings, and small Pydantic models.

## Files

```text
src/robot_tools/services/foo/
├── __init__.py                 public exports
├── contract.py                 identity, actions, requests, responses
└── client.py                   typed downstream API

runtimes/foo/
├── .python-version             runtime Python pin
├── pyproject.toml              light/fake server environment
├── uv.lock
├── real/
│   ├── pyproject.toml          real model dependencies
│   └── uv.lock
├── server.py                   HTTP handlers and fake/real selection
├── backend.py                  headless upstream adapter
├── install.sh                  exact real environment and verification
├── download.sh                 external model artifacts
├── doctor.py                   read-only real import probe
├── README.md                   service-specific operation
├── PREREQUISITES.md            host, access, storage, tested baseline
└── upstream/                   optional pinned Git submodule

tests/                          contract, client, runtime, and fake E2E tests
examples/                       minimal downstream call and tuning config
```

The remaining edits are an entry in `src/robot_tools/registry.py`, example
launch and endpoint profiles, and `.gitmodules` when the upstream is a
submodule.

## 1. Define the contract

`contract.py` is the shared source of truth for both sides of the HTTP
boundary:

```python
from typing import Literal

import numpy as np

from robot_tools.core.wire import BaseRequest, BaseResponse

SERVICE_ID = "foo"
API_VERSION = "1"
ACTIONS = frozenset({"infer"})


class InferRequest(BaseRequest):
    action: Literal["infer"] = "infer"
    image: np.ndarray
    threshold: float = 0.5


class InferResponse(BaseResponse):
    masks: np.ndarray
    scores: np.ndarray
```

- Inherit from `BaseRequest` and `BaseResponse`.
- Give every request a literal action with a default. Handler registration
  derives the HTTP route from it.
- Document array shape, dtype, coordinate frame, and units.
- Reject invalid values before they reach expensive inference.
- Increase `API_VERSION` when an existing caller can no longer interpret the
  request, response, action, or semantics.

## 2. Add the typed client

```python
class FooClient(BaseClient):
    SUPPORTED_SERVICE_APIS = {SERVICE_ID: frozenset({API_VERSION})}
    REQUIRED_ACTIONS = ACTIONS

    def infer(self, image: np.ndarray, *, threshold: float = 0.5) -> InferResponse:
        request = InferRequest(image=np.asarray(image), threshold=threshold)
        return self._request(request, InferResponse)
```

Export `FooClient` from the service's `__init__.py`. Client methods remain
synchronous and return the complete typed response. Perform cheap input
normalization and local-only parameter validation before `_request()`.

The public SDK must not import Torch, CUDA packages, the upstream model, or its
artifact tooling.

## 3. Build the runtime

The light `runtimes/foo/pyproject.toml` needs only
`robot-tools[server]` and dependencies required by fake mode. Put the full
inference graph in `runtimes/foo/real/pyproject.toml` and commit both lock
files.

The light and real projects share `runtimes/foo/.venv`:

- `robot-tools sync` restores the exact light environment;
- `robot-tools install` syncs the exact real lock, verifies it, and records
  the installed lock digest;
- a later `sync` intentionally invalidates real mode, so `install` must be
  run again.

Use a uv-managed Python when the model JIT-compiles native extensions and
needs development headers.

## 4. Separate server and backend

```python
class FooServer(BaseServer):
    @handler(InferRequest)
    def infer(self, request: InferRequest) -> InferResponse:
        if self._fake:
            return InferResponse(
                masks=np.zeros((1, *request.image.shape[:2]), dtype=bool),
                scores=np.ones((1,), dtype=np.float32),
            )
        return self.backend.infer(request)
```

- `server.py` owns HTTP handlers, startup arguments, readiness, and the fake
  branch.
- `backend.py` owns model loading, upstream preprocessing, inference, and
  output adaptation.
- Import the heavy backend only in real mode so fake mode works without the
  upstream or model dependencies.
- Put values that vary per request in the contract. Put model-loading choices
  in server startup arguments and `services.yaml`.
- Keep rendering and product-specific conversions in downstream code.

## 5. Make real deployment verifiable

| File | Responsibility |
|---|---|
| `install.sh` | Sync `real/uv.lock`, verify the dependency graph and imports, run `doctor.py`, then record real-install state |
| `download.sh` | Fetch only this service's artifacts; support external storage and safe repeated runs |
| `doctor.py` | Import the actual backend/model modules without mutating the environment |
| `PREREQUISITES.md` | Record drivers, system tools, credentials, storage variables, tested versions, and revalidation triggers |
| Runtime `README.md` | Document public actions, startup options, model locations, and real acceptance checks |

The launcher initializes the selected upstream submodule before
`install.sh`. Do not initialize other services or download checkpoints as an
import side effect.

## 6. Register and expose the service

Add one `ServiceRuntimeSpec` to `SERVICE_REGISTRY`:

```python
"foo": ServiceRuntimeSpec(
    service_id="foo",
    runtime_dir=REPO_ROOT / "runtimes" / "foo",
    server_command=("python", "server.py"),
    default_port=5560,
    gpu_required=True,
),
```

Then update:

- `services.example.yaml` with a fake entry and documented real startup
  arguments;
- `endpoints.example.yaml` with the same service ID and port;
- `examples/` with the smallest useful typed call;
- the supported-services table in the root README.

## 7. Verify both boundaries

At minimum, add tests for:

- request and response validation;
- client preprocessing and local parameter rejection;
- service identity, API version, and required actions;
- fake HTTP/MessagePack round trips without an initialized upstream;
- runtime files, locks, scripts, imports, and fake startup;
- selection: installing `foo` must not initialize another upstream.

Run the fake path first:

```bash
uv run robot-tools sync
uv run robot-tools launch
# In another terminal:
uv run robot-tools status endpoints.yaml
uv run --extra server pytest -q
```

Then verify the real path on a supported GPU host:

```bash
uv run robot-tools install
uv run robot-tools download
# Remove --fake from foo in services.yaml.
uv run robot-tools doctor
uv run robot-tools launch
```

Record a small real inference result, expected output shape and dtype, tested
hardware, and any unverified conditions in the runtime documentation.

## Definition of done

- [ ] The root SDK remains free of model dependencies.
- [ ] Contract, client, server, and backend responsibilities are separate.
- [ ] Fake mode needs no upstream source, weights, GPU, or real dependencies.
- [ ] Light and real lock files are current and committed.
- [ ] Install, download, doctor, and launch work from a clean selected profile.
- [ ] Unselected upstreams remain uninitialized.
- [ ] Registry, YAML examples, ports, docs, and public exports agree.
- [ ] Fake tests pass in CI and real acceptance evidence is recorded.
