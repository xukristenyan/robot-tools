from __future__ import annotations

import json

from robot_tools.runtime_state import (
    clear_real_install,
    real_install_marker,
    record_real_install,
    validate_real_install,
)


def _runtime(tmp_path):
    runtime_dir = tmp_path / "fastfs"
    (runtime_dir / ".venv").mkdir(parents=True)
    (runtime_dir / "real").mkdir()
    (runtime_dir / "real" / "uv.lock").write_text("locked-v1\n")
    return runtime_dir


def test_record_validate_and_clear_real_install(tmp_path):
    runtime_dir = _runtime(tmp_path)

    marker = record_real_install(runtime_dir)

    assert marker == real_install_marker(runtime_dir)
    assert validate_real_install(runtime_dir) is None
    payload = json.loads(marker.read_text())
    assert payload["service_id"] == "fastfs"
    assert len(payload["real_lock_sha256"]) == 64

    clear_real_install(runtime_dir)
    assert not marker.exists()
    assert validate_real_install(runtime_dir) == "real environment is not installed (run install)"


def test_changed_real_lock_invalidates_install_state(tmp_path):
    runtime_dir = _runtime(tmp_path)
    record_real_install(runtime_dir)

    (runtime_dir / "real" / "uv.lock").write_text("locked-v2\n")

    assert validate_real_install(runtime_dir) == "real environment does not match real/uv.lock (rerun install)"


def test_invalid_marker_is_rejected(tmp_path):
    runtime_dir = _runtime(tmp_path)
    real_install_marker(runtime_dir).write_text("not-json")

    assert validate_real_install(runtime_dir) == "real environment marker is invalid (rerun install)"
