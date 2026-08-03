# Module 4: Fake 到 Real——一套 Runtime 的一生

## Teaching Arc
- **Metaphor:** 赛车模拟器与真实赛车：先在模拟器验证方向盘和赛道，再装发动机、燃料和轮胎；检车贴证明当前配置真的通过过。
- **Opening hook:** 为什么 sync 之后 fake 能跑，而真实模型必须再 install 和 download？
- **Key insight:** 每个 runtime 有 light 与 real 两份锁定项目，共用一个 venv；services.yaml 是选择边界，marker 防止把 light 环境误认成 real。
- **Why should I care?:** 能安全换设备、只装需要的 upstream，并看懂 doctor 为什么拒绝启动。

## Code Snippets (pre-extracted)

File: src/robot_tools/launcher.py (lines 104-118)

    def _sync_one(spec: ServiceRuntimeSpec, extras: list[str]) -> None:
        # The lightweight and real projects intentionally share one .venv. Clear
        # the real marker before exact sync starts because even a failed sync may
        # have partially removed real-only packages.
        clear_real_install(spec.runtime_dir)
        cmd = ["uv", "sync", "--locked"]
        for e in extras:
            cmd.extend(["--extra", e])
        extras_str = " ".join(f"--extra {e}" for e in extras) or "(no extras)"
        logger.info("[sync] %s: uv sync --locked %s (cwd=%s)", spec.service_id, extras_str, spec.runtime_dir)
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        subprocess.run(cmd, cwd=spec.runtime_dir, env=env, check=True)

File: src/robot_tools/launcher.py (lines 308-313)

    def install(config_path: Path) -> None:
        """Initialize and install only the service runtimes selected by a profile."""
        for spec, svc in _parse_services(config_path):
            _init_upstream(spec)
            _sync_one(spec, svc.extras)
            _run_runtime_script(spec, "install.sh")

File: src/robot_tools/runtime_state.py (lines 45-50)

    def validate_real_install(runtime_dir: str | Path) -> str | None:
        """Return a diagnostic when the venv is not marked for the current lock."""
        runtime_dir = Path(runtime_dir).resolve()
        marker = real_install_marker(runtime_dir)
        if not marker.is_file():
            return "real environment is not installed (run install)"

## Interactive Elements
- [x] **Code↔English translation** — sync clearing marker, selected install, marker validation.
- [x] **Quiz** — 4 operation scenarios: sync after install; missing weights; only two services; folder moved.
- [ ] **Group chat animation**
- [x] **Data flow animation** — selected profile branches into light path and real path, then doctor and launch.
- [ ] **Drag-and-drop**
- [x] **Other** — layer toggle showing light project vs real project contents; lifecycle state machine; file-role cards for pyproject, uv.lock, install.sh, download.sh, doctor.py, upstream, compat and PREREQUISITES.

## Reference Files to Read
- interactive-elements.md → Code ↔ English Translation Blocks; Layer Toggle Demo; Message Flow / Data Flow Animation; Pattern/Feature Cards; Scenario Quiz; Glossary Tooltips
- design-system.md → Color Palette; Animations & Transitions; Module Structure
- content-philosophy.md → all content rules
- gotchas.md → full checklist

## Connections
- **Previous module:** request reached a running backend.
- **Next module:** use the repeated service/runtime template to add a new capability.
- **Tone/style notes:** Define uv, venv, lockfile, dependency, upstream, submodule, checkpoint, marker/hash and idempotent. Never imply fake validates model quality.
