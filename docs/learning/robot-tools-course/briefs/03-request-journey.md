# Module 3: 一次调用到底经过了什么

## Teaching Arc
- **Metaphor:** 一件需要验身份、按统一格式封装、签收并回执的挂号包裹。
- **Opening hook:** 你写下 client.reconstruct(left, right) 后，图像怎样跑到 GPU 又变回 NumPy 数组？
- **Key insight:** client、contract、wire、route、server handler、backend 各只负责一段，双方共用 contract。
- **Why should I care?:** 结果不对时能判断是本地输入、网络协议、server handler 还是 upstream 模型的问题。

## Code Snippets (pre-extracted)

File: src/robot_tools/services/fastfs/client.py (lines 36-42)

    req = ReconstructRequest(
        left=_as_hwc3_u8(left),
        right=_as_hwc3_u8(right),
        valid_iters=valid_iters,
        max_disp=max_disp,
    )
    return self._request(req, ReconstructResponse)

File: src/robot_tools/core/wire.py (lines 35-40)

    def pack(payload: object) -> bytes:
        return msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True)

    def unpack(raw: bytes) -> object:
        return msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)

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
- [x] **Code↔English translation** — all three snippets.
- [x] **Quiz** — 4 tracing/debug scenarios: wrong dtype before request; 422 validation; 500 backend failure; response dtype mismatch.
- [x] **Group chat animation** — User Code, FastFSClient, Contract, BaseClient, FastAPI Route, FastFSServer, Backend exchange short messages.
- [x] **Data flow animation** — image array → request model → MessagePack bytes → POST route → worker → backend → response model → bytes → typed response.
- [ ] **Drag-and-drop**
- [x] **Other** — compact HTTP status/error legend and client compatibility “passport” visual.

## Reference Files to Read
- interactive-elements.md → Code ↔ English Translation Blocks; Group Chat Animation; Message Flow / Data Flow Animation; Flow Diagrams; Multiple-Choice Quizzes; Glossary Tooltips
- design-system.md → Animations & Transitions; Module Structure; Code Block Globals
- content-philosophy.md → all content rules
- gotchas.md → full checklist

## Connections
- **Previous module:** file map introduced each layer.
- **Next module:** runtime lifecycle explains how the server and backend become runnable.
- **Tone/style notes:** Give components friendly personalities but keep real names. Define HTTP, route, handler, serialization, MessagePack, schema, thread pool, status code.
