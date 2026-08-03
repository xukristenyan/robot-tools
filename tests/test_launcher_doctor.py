from __future__ import annotations

import subprocess

import pytest

from robot_tools import launcher
from robot_tools.launcher import ServiceConfig
from robot_tools.registry import ServiceRuntimeSpec
from robot_tools.runtime_state import real_install_marker, record_real_install


def _runtime(tmp_path, *, record_real: bool = False):
    runtime_dir = tmp_path / "fastfs"
    for relative_path in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "real/pyproject.toml",
        "real/uv.lock",
        "upstream/.git",
        ".venv/bin/python",
        "doctor.py",
    ):
        path = runtime_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"test fixture: {relative_path}\n")

    if record_real:
        record_real_install(runtime_dir)

    spec = ServiceRuntimeSpec(
        service_id="fastfs",
        runtime_dir=runtime_dir,
        server_command=("python", "server.py"),
        default_port=5556,
    )
    return runtime_dir, spec


def _config(*, fake: bool) -> ServiceConfig:
    return ServiceConfig(
        service_id="fastfs",
        port=5556,
        host="127.0.0.1",
        gpu=None,
        extras=[],
        extra_args=["--fake"] if fake else [],
    )


def _completed(command, returncode=0, *, stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout="", stderr=stderr)


def test_real_doctor_rejects_light_environment_after_sync(tmp_path, monkeypatch, capsys):
    runtime_dir, spec = _runtime(tmp_path)
    commands = []
    monkeypatch.setattr(launcher, "_parse_services", lambda _path: [(spec, _config(fake=False))])

    def run(command, **_kwargs):
        commands.append(command)
        return _completed(command)

    monkeypatch.setattr(launcher.subprocess, "run", run)

    assert launcher.doctor(tmp_path / "services.yaml") == 1
    assert "real environment is not installed" in capsys.readouterr().out
    assert [str(runtime_dir / ".venv" / "bin" / "python"), str(runtime_dir / "doctor.py")] not in commands


def test_real_doctor_runs_dependency_and_service_probes(tmp_path, monkeypatch, capsys):
    runtime_dir, spec = _runtime(tmp_path, record_real=True)
    commands = []
    monkeypatch.setattr(launcher, "_parse_services", lambda _path: [(spec, _config(fake=False))])

    def run(command, **_kwargs):
        commands.append(command)
        return _completed(command)

    monkeypatch.setattr(launcher.subprocess, "run", run)

    assert launcher.doctor(tmp_path / "services.yaml") == 0
    assert "local real runtime checks passed" in capsys.readouterr().out
    assert ["uv", "pip", "check", "--python", str(runtime_dir / ".venv" / "bin" / "python")] in commands
    assert [str(runtime_dir / ".venv" / "bin" / "python"), str(runtime_dir / "doctor.py")] in commands


@pytest.mark.parametrize(
    ("failing_command", "expected_message"),
    [
        ("pip-check", "runtime dependency check failed"),
        ("real-probe", "real runtime imports failed"),
    ],
)
def test_real_doctor_reports_installed_dependency_failures(
    tmp_path,
    monkeypatch,
    capsys,
    failing_command,
    expected_message,
):
    runtime_dir, spec = _runtime(tmp_path, record_real=True)
    python = runtime_dir / ".venv" / "bin" / "python"
    monkeypatch.setattr(launcher, "_parse_services", lambda _path: [(spec, _config(fake=False))])

    def run(command, **_kwargs):
        if failing_command == "pip-check" and command[:3] == ["uv", "pip", "check"]:
            return _completed(command, 1, stderr="dependency mismatch")
        if failing_command == "real-probe" and command == [str(python), str(runtime_dir / "doctor.py")]:
            return _completed(command, 1, stderr="ModuleNotFoundError: torch")
        return _completed(command)

    monkeypatch.setattr(launcher.subprocess, "run", run)

    assert launcher.doctor(tmp_path / "services.yaml") == 1
    assert expected_message in capsys.readouterr().out


def test_fake_doctor_does_not_require_real_state_or_probe(tmp_path, monkeypatch, capsys):
    runtime_dir, spec = _runtime(tmp_path)
    (runtime_dir / "doctor.py").unlink()
    commands = []
    monkeypatch.setattr(launcher, "_parse_services", lambda _path: [(spec, _config(fake=True))])

    def run(command, **_kwargs):
        commands.append(command)
        return _completed(command)

    monkeypatch.setattr(launcher.subprocess, "run", run)

    assert launcher.doctor(tmp_path / "services.yaml") == 0
    assert "local fake runtime checks passed" in capsys.readouterr().out
    assert all("doctor.py" not in " ".join(command) for command in commands)


def test_light_sync_clears_real_install_state_before_running(tmp_path, monkeypatch):
    runtime_dir, spec = _runtime(tmp_path, record_real=True)
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **_kwargs: _completed(command))

    launcher._sync_one(spec, [])

    assert not real_install_marker(runtime_dir).exists()


def test_install_scripts_record_state_only_after_real_probe():
    runtimes_dir = launcher.REPO_ROOT / "runtimes"
    for service_id in ("fastfs", "sam3", "graspgen", "graspgenx"):
        script = (runtimes_dir / service_id / "install.sh").read_text()
        probe = '"$VENV_DIR/bin/python" "$HERE/doctor.py"'
        record = '"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"'
        assert probe in script
        assert record in script
        assert script.index(probe) < script.index(record)
