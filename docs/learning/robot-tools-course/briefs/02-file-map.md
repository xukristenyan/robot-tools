# Module 2: 文件地图——遇到需求先去哪一层

## Teaching Arc
- **Metaphor:** 一座有清晰标签的工具仓库：前台工具、公共规则、独立机器房、说明书和质检区各有货架。
- **Opening hook:** 你已经知道大结构，现在要做到看到一个需求就能马上判断应该打开哪类文件。
- **Key insight:** src 是轻量公共 SDK，runtimes 是重型部署环境；重复出现的文件名是一套固定模板。
- **Why should I care?:** 给 AI 下指令时可以准确说“改 contract，不要把 Torch 放进根 pyproject”，避免跨层污染。

## Code Snippets (pre-extracted)

File: src/robot_tools/registry.py (lines 17-31)

    @dataclass(frozen=True)
    class ServiceRuntimeSpec:
        service_id: str
        runtime_dir: Path
        server_command: tuple[str, ...]  # command after ``uv run``
        default_port: int
        gpu_required: bool = True

    SERVICE_REGISTRY: dict[str, ServiceRuntimeSpec] = {
        "graspgen": ServiceRuntimeSpec(
            service_id="graspgen",
            runtime_dir=REPO_ROOT / "runtimes" / "graspgen",
            server_command=("python", "server.py"),
            default_port=5557,

File: endpoints.example.yaml (lines 17-21)

    endpoints:
      graspgen:  { host: 127.0.0.1, port: 5557 }
      graspgenx: { host: 127.0.0.1, port: 5559 }
      fastfs:    { host: 127.0.0.1, port: 5556 }
      sam3:      { host: 127.0.0.1, port: 5558 }

File: pyproject.toml (dependency split excerpt)

    dependencies = [
        "numpy>=1.26",
        "msgpack>=1.0",
        "msgpack-numpy>=0.4.8",
        "pyyaml>=6.0",
        "pydantic>=2",
        "typing-extensions>=4",
        "httpx>=0.27",
    ]

## Interactive Elements
- [x] **Code↔English translation** — registry and endpoint mappings; root dependency list.
- [x] **Quiz** — 4 “where would you change it?” scenarios: new request field; CUDA version; client host; new troubleshooting doc.
- [ ] **Group chat animation**
- [ ] **Data flow animation**
- [x] **Drag-and-drop** — match file/task to SDK, service API, runtime, configuration, examples/tests/docs.
- [x] **Other** — large visual file tree covering root files, core files, each service’s init/client/contract, each runtime’s README/PREREQUISITES/pyproject/real/server/backend/install/download/doctor/compat/upstream, examples, tests and docs.

## Reference Files to Read
- interactive-elements.md → Code ↔ English Translation Blocks; Drag-and-Drop Matching; Visual File Tree; Pattern/Feature Cards; Multiple-Choice Quizzes; Glossary Tooltips
- design-system.md → Spacing & Layout; Module Structure; Responsive Breakpoints
- content-philosophy.md → all content rules
- gotchas.md → full checklist

## Connections
- **Previous module:** first run commands and one-service install.
- **Next module:** follow one real request through the file map.
- **Tone/style notes:** Do not re-teach the overall architecture at length. Make this a searchable mental index. Define package, module, registry, lockfile, adapter, fixture and editable install.
