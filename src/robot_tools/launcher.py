"""Manage service runtimes declared in a YAML profile.

Usage:
    robot-tools sync [profile]      # exact lightweight/fake environments
    robot-tools install [profile]   # selected upstreams + real dependencies
    robot-tools download [profile]  # selected model artifacts
    robot-tools doctor [profile]    # read-only local checks
    robot-tools launch [profile]    # start each selected HTTP server
    robot-tools status endpoints.yaml   # client-side compatibility check of each endpoint
    robot-tools list                    # show registered service runtimes

`profile` defaults to `services.yaml`. This launch profile lives on the machine that runs
the runtimes. Clients connect via a separate `endpoints.yaml` (see
`robot_tools.core.endpoints`), so moving a service to another machine never touches
this file.

services.yaml:
    services:
      - name: graspgen
        port: 5557                # optional, registry default
        host: 127.0.0.1           # optional; interface to serve on, as in
                                  #   uvicorn --host; 0.0.0.0 is for a trusted,
                                  #   firewalled LAN only (no built-in TLS/auth)
        gpu: 0                    # optional, sets CUDA_VISIBLE_DEVICES
        extras: []                # optional, uv extras used by sync/install
        extra_args: ["--fake"]    # optional server CLI arguments
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from robot_tools.core.exceptions import ProtocolMismatchError
from robot_tools.registry import REPO_ROOT, SERVICE_REGISTRY, ServiceRuntimeSpec, get_service_runtime
from robot_tools.runtime_state import clear_real_install, validate_real_install

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceConfig:
    service_id: str
    port: int
    host: str  # interface the server listens on (uvicorn --host semantics)
    gpu: int | None
    extras: list[str]
    extra_args: list[str]


def _parse_services(config_path: Path) -> list[tuple[ServiceRuntimeSpec, ServiceConfig]]:
    cfg = yaml.safe_load(config_path.read_text()) or {}
    raw: list[dict[str, Any]] = cfg.get("services", [])
    if not raw:
        raise ValueError(f"no `services:` entries in {config_path}")

    out: list[tuple[ServiceRuntimeSpec, ServiceConfig]] = []
    for entry in raw:
        service_id = entry["name"]
        spec = get_service_runtime(service_id)
        if not spec.runtime_dir.is_dir():
            raise FileNotFoundError(
                f"runtime directory not found: {spec.runtime_dir} — runtime management "
                "must run from a robot-tools checkout (the pip-installed SDK alone "
                "cannot manage runtimes; clients only need `endpoints.yaml` + `status`)."
            )
        out.append(
            (
                spec,
                ServiceConfig(
                    service_id=service_id,
                    port=entry.get("port", spec.default_port),
                    host=entry.get("host", "127.0.0.1"),
                    gpu=entry.get("gpu"),
                    extras=list(entry.get("extras", [])),
                    extra_args=list(entry.get("extra_args", [])),
                ),
            )
        )
    return out


# ---------- sync ----------


def sync(config_path: Path) -> None:
    """Create exact locked lightweight environments for selected services."""
    services = _parse_services(config_path)
    for spec, svc in services:
        _sync_one(spec, svc.extras)


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
    # The launcher is normally invoked through the root project's `uv run`,
    # whose VIRTUAL_ENV must not leak into a different runtime project.
    env.pop("VIRTUAL_ENV", None)
    subprocess.run(cmd, cwd=spec.runtime_dir, env=env, check=True)


# ---------- launch ----------


def launch(config_path: Path) -> None:
    services = _parse_services(config_path)
    _warn_shared_gpus(services)

    # Auto-sync any runtime missing its .venv.
    for spec, svc in services:
        if not (spec.runtime_dir / ".venv").exists():
            logger.info("[auto-sync] %s has no .venv, running sync first", spec.service_id)
            _sync_one(spec, svc.extras)

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        for spec, svc in services:
            procs.append((svc.service_id, _spawn(spec, svc)))
        logger.info("launched %d server process(es); Ctrl-C to stop", len(procs))
        _wait(procs)
    finally:
        _shutdown(procs)


def _warn_shared_gpus(
    services: list[tuple[ServiceRuntimeSpec, ServiceConfig]],
) -> None:
    groups: dict[str, list[str]] = {}
    for spec, svc in services:
        if not spec.gpu_required:
            continue
        gpu = str(svc.gpu) if svc.gpu is not None else "default"
        groups.setdefault(gpu, []).append(svc.service_id)

    for gpu, names in groups.items():
        if len(names) > 1:
            logger.warning(
                "services %s share GPU %s; cross-service concurrency is "
                "allowed, so verify peak VRAM and concurrent latency",
                ", ".join(names),
                gpu,
            )


def _spawn(spec: ServiceRuntimeSpec, svc: ServiceConfig) -> subprocess.Popen:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    if svc.gpu is not None and spec.gpu_required:
        env["CUDA_VISIBLE_DEVICES"] = str(svc.gpu)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(spec.runtime_dir), env.get("PYTHONPATH", "")]))

    # --no-sync keeps launch read-only and preserves real dependencies installed
    # by install.sh. Explicit `sync` intentionally restores the locked light env.
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(spec.runtime_dir),
        *spec.server_command,
        "--host",
        svc.host,
        "--port",
        str(svc.port),
        *svc.extra_args,
    ]
    logger.info(
        "starting %s: %s (CUDA_VISIBLE_DEVICES=%s)",
        svc.service_id,
        " ".join(cmd),
        env.get("CUDA_VISIBLE_DEVICES", ""),
    )
    return subprocess.Popen(cmd, env=env, cwd=spec.runtime_dir)


def _wait(procs: list[tuple[str, subprocess.Popen]]) -> None:
    while procs:
        for name, p in procs:
            rc = p.poll()
            if rc is not None:
                logger.error("%s exited with code %s", name, rc)
                return
        time.sleep(1.0)


def _shutdown(procs: list[tuple[str, subprocess.Popen]], grace: float = 5.0) -> None:
    for name, p in procs:
        if p.poll() is None:
            logger.info("terminating %s (pid=%d)", name, p.pid)
            p.terminate()
    deadline = time.monotonic() + grace
    for _name, p in procs:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()


# ---------- status ----------


def _status_client_types():
    """Return typed clients keyed by the service identities they accept."""
    from robot_tools.services.fastfs import FastFSClient
    from robot_tools.services.graspgen import GraspGenClient
    from robot_tools.services.graspgenx import GraspGenXClient
    from robot_tools.services.sam3 import SAM3Client

    client_types = (FastFSClient, GraspGenClient, GraspGenXClient, SAM3Client)
    return {
        service_id: client_type for client_type in client_types for service_id in client_type.SUPPORTED_SERVICE_APIS
    }


def status(endpoints_path: Path, timeout_ms: int = 2000) -> int:
    """Compatibility-check every service in endpoints.yaml from the client side.

    Returns a process exit code: 0 if all endpoints are compatible, 1 otherwise.
    """
    from robot_tools import __version__
    from robot_tools.core.endpoints import load_endpoints

    n_down = 0
    client_types = _status_client_types()
    for ep in load_endpoints(endpoints_path).values():
        client_type = client_types.get(ep.service_id)
        if client_type is None:
            print(f"{ep.service_id:12s} ✗ incompatible: no typed client is registered")
            n_down += 1
            continue

        client = client_type(host=ep.host, port=ep.port, timeout_ms=timeout_ms, wait=False)
        t0 = time.monotonic()
        try:
            info = client.check_compatibility(timeout_s=timeout_ms / 1000)
            dt_ms = (time.monotonic() - t0) * 1000
            package_version = info.get("package_version", "?")
            drift = (
                ""
                if package_version == __version__
                else f"  [server package v{package_version} != client v{__version__}]"
            )
            print(
                f"{ep.service_id:12s} ✓ {ep.host}:{ep.port} "
                f"({dt_ms:.0f}ms, service={info.get('service_id', '?')}, "
                f"wire={info.get('wire_protocol_version', '?')}, "
                f"api={info.get('api_version', '?')}, "
                f"package={package_version}, "
                f"actions={','.join(info.get('actions', []))}){drift}"
            )
        except ProtocolMismatchError as exc:
            print(f"{ep.service_id:12s} ✗ {ep.host}:{ep.port} incompatible: {exc}")
            n_down += 1
        except (httpx.HTTPError, OSError) as exc:
            print(f"{ep.service_id:12s} ✗ {ep.host}:{ep.port} unreachable: {exc}")
            n_down += 1
        finally:
            client.close()
    return 1 if n_down else 0


# ---------- install / download / doctor ----------


def _run_runtime_script(spec: ServiceRuntimeSpec, script: str) -> None:
    """Run an optional runtime-local lifecycle script."""
    path = spec.runtime_dir / script
    if not path.exists():
        logger.info("[%s] no %s; nothing to do", spec.service_id, script)
        return
    logger.info("[%s] running %s (cwd=%s)", spec.service_id, script, spec.runtime_dir)
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    subprocess.run(["bash", str(path)], cwd=spec.runtime_dir, env=env, check=True)


def _init_upstream(spec: ServiceRuntimeSpec) -> None:
    upstream = spec.runtime_dir / "upstream"
    relative = upstream.relative_to(REPO_ROOT)
    logger.info("[%s] initializing only %s", spec.service_id, relative)
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive", "--", str(relative)],
        cwd=REPO_ROOT,
        check=True,
    )


def install(config_path: Path) -> None:
    """Initialize and install only the service runtimes selected by a profile."""
    for spec, svc in _parse_services(config_path):
        _init_upstream(spec)
        _sync_one(spec, svc.extras)
        _run_runtime_script(spec, "install.sh")


def download(config_path: Path) -> None:
    """Download artifacts only for service runtimes selected by a profile."""
    for spec, _svc in _parse_services(config_path):
        _run_runtime_script(spec, "download.sh")


def doctor(config_path: Path) -> int:
    """Run read-only checks for each service runtime selected by a profile."""
    failures = 0
    for spec, svc in _parse_services(config_path):
        real_mode = "--fake" not in svc.extra_args
        problems: list[str] = []
        for filename in ("pyproject.toml", "uv.lock", ".python-version"):
            if not (spec.runtime_dir / filename).is_file():
                problems.append(f"missing {filename}")

        python = spec.runtime_dir / ".venv" / "bin" / "python"
        if not python.is_file():
            problems.append("missing .venv (run sync or install)")
        else:
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            probe = subprocess.run(
                [str(python), "-c", "import fastapi, robot_tools, uvicorn"],
                cwd=spec.runtime_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode:
                detail = (probe.stderr or probe.stdout).strip().splitlines()
                problems.append(f"runtime imports failed: {detail[-1] if detail else 'unknown error'}")

            pip_check = subprocess.run(
                ["uv", "pip", "check", "--python", str(python)],
                cwd=spec.runtime_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if pip_check.returncode:
                detail = (pip_check.stderr or pip_check.stdout).strip().splitlines()
                problems.append(f"runtime dependency check failed: {detail[-1] if detail else 'unknown error'}")

        lock_check = subprocess.run(
            ["uv", "lock", "--check"],
            cwd=spec.runtime_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if lock_check.returncode:
            problems.append("uv.lock is stale")

        if real_mode:
            upstream = spec.runtime_dir / "upstream"
            if not (upstream / ".git").exists():
                problems.append("upstream submodule is not initialized (run install)")

            real_project = spec.runtime_dir / "real"
            if real_project.is_dir():
                for filename in ("pyproject.toml", "uv.lock"):
                    if not (real_project / filename).is_file():
                        problems.append(f"missing real/{filename}")
                if (real_project / "uv.lock").is_file():
                    real_lock_check = subprocess.run(
                        ["uv", "lock", "--check", "--project", str(real_project)],
                        cwd=spec.runtime_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if real_lock_check.returncode:
                        problems.append("real/uv.lock is stale")

            state_problem = validate_real_install(spec.runtime_dir)
            if state_problem:
                problems.append(state_problem)

            real_doctor = spec.runtime_dir / "doctor.py"
            if not real_doctor.is_file():
                problems.append("missing doctor.py real-runtime probe")
            elif python.is_file() and state_problem is None:
                real_probe = subprocess.run(
                    [str(python), str(real_doctor)],
                    cwd=spec.runtime_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if real_probe.returncode:
                    detail = (real_probe.stderr or real_probe.stdout).strip().splitlines()
                    problems.append(f"real runtime imports failed: {detail[-1] if detail else 'unknown error'}")

        doctor_script = spec.runtime_dir / "doctor.sh"
        if doctor_script.exists():
            check = subprocess.run(
                ["bash", str(doctor_script)],
                cwd=spec.runtime_dir,
                check=False,
            )
            if check.returncode:
                problems.append("service-specific doctor failed")

        if problems:
            failures += 1
            print(f"{spec.service_id:12s} ✗ {'; '.join(problems)}")
        else:
            mode = "real" if real_mode else "fake"
            print(f"{spec.service_id:12s} ✓ local {mode} runtime checks passed")
    return 1 if failures else 0


# ---------- CLI ----------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-request INFO lines
    parser = argparse.ArgumentParser(prog="robot-tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="Sync exact light environments from services.yaml")
    p_sync.add_argument("config", nargs="?", type=Path, default=Path("services.yaml"))

    p_install = sub.add_parser("install", help="Install selected upstreams and real dependencies")
    p_install.add_argument("config", nargs="?", type=Path, default=Path("services.yaml"))

    p_download = sub.add_parser("download", help="Download artifacts for selected services")
    p_download.add_argument("config", nargs="?", type=Path, default=Path("services.yaml"))

    p_doctor = sub.add_parser("doctor", help="Check selected local runtimes without changing them")
    p_doctor.add_argument("config", nargs="?", type=Path, default=Path("services.yaml"))

    p_launch = sub.add_parser("launch", help="Start selected HTTP servers; auto-sync missing venvs")
    p_launch.add_argument("config", nargs="?", type=Path, default=Path("services.yaml"))

    p_status = sub.add_parser("status", help="Compatibility-check each service in endpoints.yaml")
    p_status.add_argument("config", type=Path, help="Path to endpoints.yaml")
    p_status.add_argument("--timeout-ms", type=int, default=2000)

    sub.add_parser("list", help="List registered service runtimes")

    args = parser.parse_args()

    if args.cmd == "sync":
        sync(args.config)
    elif args.cmd == "install":
        install(args.config)
    elif args.cmd == "download":
        download(args.config)
    elif args.cmd == "doctor":
        sys.exit(doctor(args.config))
    elif args.cmd == "status":
        sys.exit(status(args.config, timeout_ms=args.timeout_ms))
    elif args.cmd == "launch":
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        launch(args.config)
    elif args.cmd == "list":
        for service_id, spec in SERVICE_REGISTRY.items():
            print(f"{service_id:12s} port={spec.default_port:<6d} runtime={spec.runtime_dir}")
