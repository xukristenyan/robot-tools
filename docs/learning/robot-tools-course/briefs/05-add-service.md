# Module 5: 亲手增加一个新 Service

## Teaching Arc
- **Metaphor:** 给一套积木系统设计官方扩展包：接口尺寸先固定，再做控制器、机器主体、动力模块和质检说明。
- **Opening hook:** 假设明天要接入一个名为 foo 的新视觉模型，你应该按什么顺序创建文件？
- **Key insight:** contract 先定义稳定 API，client 和 server 共用它；runtime 隔离重依赖；registry 只负责让 launcher 找到它。
- **Why should I care?:** 能把新增模型拆成可检查的小步骤，并给 AI 清晰的验收标准，而不是让它到处复制代码。

## Code Snippets (pre-extracted)

File: src/robot_tools/services/fastfs/contract.py (lines 9-24)

    SERVICE_ID = "fastfs"
    API_VERSION = "1"
    ACTIONS = frozenset({"reconstruct"})

    class ReconstructRequest(BaseRequest):
        action: Literal["reconstruct"] = "reconstruct"
        left: np.ndarray  # (H, W, 3) uint8 or (H, W) gray; client coerces to HxWx3 uint8
        right: np.ndarray  # same shape as left
        # Per-call refinement knobs; None => use server's startup defaults.
        valid_iters: int | None = None
        max_disp: int | None = None

    class ReconstructResponse(BaseResponse):
        disparity: np.ndarray  # (H, W) float32, in pixels — depth = K_fx * baseline / disp

File: src/robot_tools/services/fastfs/client.py (lines 17-19)

    class FastFSClient(BaseClient):
        SUPPORTED_SERVICE_APIS: ClassVar[dict[str, frozenset[str]]] = {SERVICE_ID: frozenset({API_VERSION})}
        REQUIRED_ACTIONS = ACTIONS

File: runtimes/fastfs/server.py (lines 64-76)

    @handler(ReconstructRequest)
    def reconstruct(self, req: ReconstructRequest) -> ReconstructResponse:
        if self._fake:
            H, W = req.left.shape[:2]
            return ReconstructResponse(disparity=np.zeros((H, W), dtype=np.float32))

        disp, _timings = self.backend.infer(
            req.left,
            req.right,
            valid_iters=req.valid_iters,
            max_disp=req.max_disp,
        )
        return ReconstructResponse(disparity=disp)

## Interactive Elements
- [x] **Code↔English translation** — contract identity/action, client compatibility declaration, server fake/real branch.
- [x] **Quiz** — 5 architecture scenarios: where to put camera conversion; startup vs per-call option; missing upstream import; API bump; port collision.
- [ ] **Group chat animation**
- [x] **Data flow animation** — seven-step construction path: submodule → contract → client/export → runtime files → registry → configs/docs → fake/real tests.
- [x] **Drag-and-drop** — sort proposed code into contract, client, server, backend, install/download, registry or downstream.
- [x] **Other** — “definition of done” checklist and reusable AI prompt template for adding a service.

## Reference Files to Read
- interactive-elements.md → Code ↔ English Translation Blocks; Drag-and-Drop Matching; Numbered Step Cards; Message Flow / Data Flow Animation; Scenario Quiz; Callout Boxes; Glossary Tooltips
- design-system.md → Spacing & Layout; Module Structure; Responsive Breakpoints
- content-philosophy.md → all content rules
- gotchas.md → full checklist

## Connections
- **Previous module:** runtime files and lifecycle are now familiar.
- **Next module:** operate and debug the new service with status, doctor, logs and tests.
- **Tone/style notes:** This is the practical capstone. Be explicit about __init__.py, registry, example YAML, executable scripts, fake tests, real probes, docs and versioning. Define decorator, Literal, adapter, invariant and acceptance test.
