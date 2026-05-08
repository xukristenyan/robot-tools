"""Launch & sync model servers declared in a YAML config.

Usage:
    robot-tools sync services.yaml      # uv sync each listed server's venv
    robot-tools launch services.yaml    # start each listed server (auto-sync if .venv missing)
    robot-tools list                    # show registered servers

services.yaml:
    services:
      - name: graspgen
        port: 5557                # optional, default from registry
        host: 127.0.0.1           # optional
        gpu: 0                    # optional, sets CUDA_VISIBLE_DEVICES
        extras: [real]            # optional, uv extras to install via `uv sync --extra ...`
        extra_args: ["--mock"]    # optional, CLI args appended to the server command
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

import yaml

from robot_tools.registry import SERVER_REGISTRY, ServerSpec, get

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceConfig:
    name: str
    port: int
    host: str
    gpu: int | None
    extras: list[str]
    extra_args: list[str]


def _parse_services(config_path: Path) -> list[tuple[ServerSpec, ServiceConfig]]:
    cfg = yaml.safe_load(config_path.read_text())
    raw: list[dict[str, Any]] = cfg.get("services", [])
    if not raw:
        raise ValueError(f"no `services:` entries in {config_path}")

    out: list[tuple[ServerSpec, ServiceConfig]] = []
    for entry in raw:
        name = entry["name"]
        spec = get(name)
        out.append((spec, ServiceConfig(
            name=name,
            port=entry.get("port", spec.default_port),
            host=entry.get("host", "127.0.0.1"),
            gpu=entry.get("gpu"),
            extras=list(entry.get("extras", [])),
            extra_args=list(entry.get("extra_args", [])),
        )))
    return out


# ---------- sync ----------

def sync(config_path: Path) -> None:
    """uv sync each listed server's venv with its declared extras."""
    services = _parse_services(config_path)
    for spec, svc in services:
        _sync_one(spec, svc.extras)


def _sync_one(spec: ServerSpec, extras: list[str]) -> None:
    # --inexact: don't remove packages not declared in pyproject.toml. Servers
    # like graspgen install heavy ML deps via `uv pip install -e ./upstream`
    # (their setup.sh) — those would be wiped by a strict `uv sync`.
    cmd = ["uv", "sync", "--inexact"]
    for e in extras:
        cmd.extend(["--extra", e])
    extras_str = " ".join(f"--extra {e}" for e in extras) or "(no extras)"
    logger.info("[sync] %s: uv sync --inexact %s (cwd=%s)", spec.name, extras_str, spec.project_dir)
    subprocess.run(cmd, cwd=spec.project_dir, check=True)


# ---------- launch ----------

def launch(config_path: Path) -> None:
    services = _parse_services(config_path)

    # Auto-sync any server missing its .venv
    for spec, svc in services:
        if not (spec.project_dir / ".venv").exists():
            logger.info("[auto-sync] %s has no .venv, running sync first", spec.name)
            _sync_one(spec, svc.extras)

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        for spec, svc in services:
            procs.append((svc.name, _spawn(spec, svc)))
        logger.info("launched %d server(s); Ctrl-C to stop", len(procs))
        _wait(procs)
    finally:
        _shutdown(procs)


def _spawn(spec: ServerSpec, svc: ServiceConfig) -> subprocess.Popen:
    env = os.environ.copy()
    if svc.gpu is not None and spec.gpu_required:
        env["CUDA_VISIBLE_DEVICES"] = str(svc.gpu)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(spec.project_dir), env.get("PYTHONPATH", "")]))

    # --no-sync: we already synced via _sync_one (with --inexact). Letting
    # `uv run` re-sync would strip pip-installed packages (grasp_gen etc.).
    cmd = [
        "uv", "run", "--no-sync", "--project", str(spec.project_dir),
        *spec.entry,
        "--host", svc.host,
        "--port", str(svc.port),
        *svc.extra_args,
    ]
    logger.info("starting %s: %s (CUDA_VISIBLE_DEVICES=%s)",
                svc.name, " ".join(cmd), env.get("CUDA_VISIBLE_DEVICES", ""))
    return subprocess.Popen(cmd, env=env, cwd=spec.project_dir)


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


# ---------- setup / download (one-shot, per-server scripts) ----------

def _run_server_script(name: str, script: str) -> None:
    """Run a bash script located in the server's project_dir, if it exists."""
    spec = get(name)
    path = spec.project_dir / script
    if not path.exists():
        logger.info("[%s] no %s; nothing to do", name, script)
        return
    logger.info("[%s] running %s (cwd=%s)", name, script, spec.project_dir)
    subprocess.run(["bash", str(path)], cwd=spec.project_dir, check=True)


def setup(name: str) -> None:
    """Run servers/<name>/setup.sh — for manual install steps uv can't express."""
    _run_server_script(name, "setup.sh")


def download(name: str) -> None:
    """Run servers/<name>/download.sh — for fetching model weights."""
    _run_server_script(name, "download.sh")


# ---------- CLI ----------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="robot-tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="uv sync each server listed in services.yaml")
    p_sync.add_argument("config", type=Path)

    p_launch = sub.add_parser("launch", help="Start servers; auto-sync missing venvs")
    p_launch.add_argument("config", type=Path)

    p_setup = sub.add_parser("setup", help="Run a server's setup.sh (manual install steps)")
    p_setup.add_argument("server", help="Server name (see `robot-tools list`)")

    p_download = sub.add_parser("download", help="Run a server's download.sh (fetch weights)")
    p_download.add_argument("server", help="Server name (see `robot-tools list`)")

    sub.add_parser("list", help="List registered servers")

    args = parser.parse_args()

    if args.cmd == "sync":
        sync(args.config)
    elif args.cmd == "launch":
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        launch(args.config)
    elif args.cmd == "setup":
        setup(args.server)
    elif args.cmd == "download":
        download(args.server)
    elif args.cmd == "list":
        for name, spec in SERVER_REGISTRY.items():
            print(f"{name:12s} port={spec.default_port:<6d} project={spec.project_dir}")
