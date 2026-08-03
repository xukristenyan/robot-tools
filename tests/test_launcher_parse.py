import logging
import sys
from pathlib import Path

import pytest

from robot_tools import launcher
from robot_tools.launcher import _parse_services, _warn_shared_gpus
from robot_tools.registry import get_service_runtime


def _write(tmp_path, text):
    p = tmp_path / "services.yaml"
    p.write_text(text)
    return p


def test_host_defaults_to_loopback(tmp_path):
    p = _write(tmp_path, "services:\n  - name: fastfs\n")
    [(spec, svc)] = _parse_services(p)
    assert svc.host == "127.0.0.1"
    assert svc.port == spec.default_port


def test_host_explicit(tmp_path):
    p = _write(tmp_path, "services:\n  - name: fastfs\n    host: 0.0.0.0\n    port: 6000\n")
    [(_, svc)] = _parse_services(p)
    assert svc.host == "0.0.0.0"
    assert svc.port == 6000


def test_unknown_service_rejected(tmp_path):
    p = _write(tmp_path, "services:\n  - name: nonexistent\n")
    with pytest.raises(KeyError, match="nonexistent"):
        _parse_services(p)


def test_registry_points_at_runtime_directories():
    spec = get_service_runtime("sam3")
    assert spec.service_id == "sam3"
    assert spec.runtime_dir.parts[-2:] == ("runtimes", "sam3")
    assert spec.server_command == ("python", "server.py")


@pytest.mark.parametrize(
    ("command", "function_name", "exits"),
    [
        ("sync", "sync", False),
        ("install", "install", False),
        ("download", "download", False),
        ("doctor", "doctor", True),
        ("launch", "launch", False),
    ],
)
def test_lifecycle_cli_defaults_to_services_yaml(monkeypatch, command, function_name, exits):
    configs = []

    def run(config):
        configs.append(config)
        return 0

    monkeypatch.setattr(sys, "argv", ["robot-tools", command])
    monkeypatch.setattr(launcher, function_name, run)
    monkeypatch.setattr(launcher.signal, "signal", lambda *_args: None)

    if exits:
        with pytest.raises(SystemExit) as exc_info:
            launcher.main()
        assert exc_info.value.code == 0
    else:
        launcher.main()

    assert configs == [Path("services.yaml")]


def test_sync_is_exact_and_locked(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(launcher.subprocess, "run", run)
    spec = get_service_runtime("sam3")
    launcher._sync_one(spec, ["example"])

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == ["uv", "sync", "--locked", "--extra", "example"]
    assert kwargs["cwd"] == spec.runtime_dir
    assert kwargs["check"] is True
    assert "VIRTUAL_ENV" not in kwargs["env"]


def test_install_initializes_only_services_in_profile(tmp_path, monkeypatch):
    profile = _write(tmp_path, "services:\n  - name: sam3\n  - name: fastfs\n")
    initialized = []
    synced = []
    scripts = []
    monkeypatch.setattr(launcher, "_init_upstream", lambda spec: initialized.append(spec.service_id))
    monkeypatch.setattr(launcher, "_sync_one", lambda spec, extras: synced.append((spec.service_id, extras)))
    monkeypatch.setattr(
        launcher,
        "_run_runtime_script",
        lambda spec, script: scripts.append((spec.service_id, script)),
    )

    launcher.install(Path(profile))

    assert initialized == ["sam3", "fastfs"]
    assert synced == [("sam3", []), ("fastfs", [])]
    assert scripts == [("sam3", "install.sh"), ("fastfs", "install.sh")]


def test_upstream_init_is_scoped_to_one_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    spec = get_service_runtime("sam3")

    launcher._init_upstream(spec)

    assert calls == [
        (
            [
                "git",
                "submodule",
                "update",
                "--init",
                "--recursive",
                "--",
                "runtimes/sam3/upstream",
            ],
            {"cwd": launcher.REPO_ROOT, "check": True},
        )
    ]


def test_download_runs_only_selected_runtime_scripts(tmp_path, monkeypatch):
    profile = _write(tmp_path, "services:\n  - name: graspgen\n  - name: fastfs\n")
    scripts = []
    monkeypatch.setattr(
        launcher,
        "_run_runtime_script",
        lambda spec, script: scripts.append((spec.service_id, script)),
    )

    launcher.download(profile)

    assert scripts == [("graspgen", "download.sh"), ("fastfs", "download.sh")]


def test_runtime_script_does_not_inherit_root_virtual_env(monkeypatch):
    calls = []
    monkeypatch.setenv("VIRTUAL_ENV", "/root-project/.venv")
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    spec = get_service_runtime("fastfs")

    launcher._run_runtime_script(spec, "install.sh")

    [(command, kwargs)] = calls
    assert command == ["bash", str(spec.runtime_dir / "install.sh")]
    assert kwargs["cwd"] == spec.runtime_dir
    assert kwargs["check"] is True
    assert "VIRTUAL_ENV" not in kwargs["env"]


def test_shared_gpu_warns_without_blocking(tmp_path, caplog):
    p = _write(
        tmp_path,
        "services:\n  - { name: sam3, gpu: 0 }\n  - { name: fastfs, gpu: 0 }\n",
    )
    services = _parse_services(p)
    with caplog.at_level(logging.WARNING):
        _warn_shared_gpus(services)
    assert "sam3, fastfs share GPU 0" in caplog.text
