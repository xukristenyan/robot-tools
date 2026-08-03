# Module 1: 从 Clone 到第一次调用

## Teaching Arc
- **Metaphor:** 出发旅行前按目的地打包：普通 clone 是拿空行李箱，services.yaml 决定只装哪几件装备。
- **Opening hook:** 假设你刚换了一台 GPU 电脑，只想运行 SAM3，而不是下载另外三个模型。
- **Key insight:** 先用 fake 模式证明整条链路正确，再单独安装和下载选中的真实模型。
- **Why should I care?:** 能避免无关 upstream、权重与 CUDA 依赖，也能把“安装失败”和“业务调用失败”分开调试。

## Code Snippets (pre-extracted)

File: README.md (lines 52-68)

    git clone <robot-tools-url>
    cd robot-tools

    services:
      - name: sam3
        port: 5558
        extra_args: ["--fake"]

File: README.md (Quick start and Run a real model)

    uv run robot-tools sync      # exact locked light/fake venvs; no upstreams or weights
    uv run robot-tools install   # selected upstreams and locked real dependencies
    uv run robot-tools download  # selected model artifacts
    uv run robot-tools doctor    # read-only local checks
    uv run robot-tools launch    # selected HTTP servers

File: README.md (lines 143-155)

    uv add --editable /path/to/robot-tools
    cp /path/to/robot-tools/endpoints.example.yaml endpoints.yaml
    uv run robot-tools status endpoints.yaml

    from robot_tools.core.endpoints import get_endpoint
    from robot_tools.services.sam3 import SAM3Client

    client = SAM3Client.from_endpoint(get_endpoint("sam3", "endpoints.yaml"))

## Interactive Elements
- [x] **Code↔English translation** — explain normal clone, one-service fake profile, lifecycle commands, downstream install.
- [x] **Quiz** — 4 scenario questions: only SAM3; fake succeeds but real fails; downstream laptop installation; moving server to another host.
- [ ] **Group chat animation**
- [x] **Data flow animation** — user clicks through clone → choose profile → sync → launch fake → status → client call; then optional install/download/doctor for real.
- [ ] **Drag-and-drop**
- [x] **Other** — numbered “first 15 minutes” checklist and fake-vs-real badges.

## Reference Files to Read
- interactive-elements.md → Code ↔ English Translation Blocks; Message Flow / Data Flow Animation; Multiple-Choice Quizzes; Numbered Step Cards; Permission/Config Badges; Glossary Tooltips
- design-system.md → Module Structure; Responsive Breakpoints; Code Block Globals
- content-philosophy.md → all content rules
- gotchas.md → full checklist

## Connections
- **Previous module:** none; open with a fresh-machine user journey.
- **Next module:** 文件地图 — explain where each command and configuration lives.
- **Tone/style notes:** Chinese, high-school level, practical and reassuring. Teal accent. Define CLI, SDK, upstream, submodule, venv, endpoint, fake mode on first use.
