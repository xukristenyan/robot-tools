from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test job
    import tomli as tomllib

import pytest

from robot_tools.registry import SERVICE_REGISTRY, ServiceRuntimeSpec


@pytest.fixture(params=sorted(SERVICE_REGISTRY), ids=lambda service_id: service_id)
def runtime_spec(request) -> ServiceRuntimeSpec:
    return SERVICE_REGISTRY[request.param]


def _load_toml(path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def test_registered_runtime_has_standard_service_layout(runtime_spec):
    runtime_dir = runtime_spec.runtime_dir
    required_paths = (
        ".python-version",
        "pyproject.toml",
        "uv.lock",
        "server.py",
        "backend.py",
        "doctor.py",
        "install.sh",
        "download.sh",
        "real/pyproject.toml",
        "real/uv.lock",
    )

    assert runtime_dir.name == runtime_spec.service_id
    assert runtime_spec.server_command == ("python", "server.py")
    for relative_path in required_paths:
        assert (runtime_dir / relative_path).is_file(), f"{runtime_spec.service_id} is missing {relative_path}"


def test_light_runtime_stays_model_free(runtime_spec):
    project = _load_toml(runtime_spec.runtime_dir / "pyproject.toml")

    assert project["project"]["dependencies"] == ["robot-tools[server]"]
    assert set(project["tool"]["uv"]["sources"]) == {"robot-tools"}
    assert "override-dependencies" not in project["tool"]["uv"]


def test_install_uses_locked_real_environment_and_records_verified_state(runtime_spec):
    script = (runtime_spec.runtime_dir / "install.sh").read_text()
    probe = '"$VENV_DIR/bin/python" "$HERE/doctor.py"'
    record = '"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"'

    assert 'UV_PROJECT_ENVIRONMENT="$VENV_DIR"' in script
    assert 'uv sync --project "$REAL_PROJECT" --locked' in script
    assert "uv pip install" not in script
    assert probe in script
    assert record in script
    assert script.index(probe) < script.index(record)
