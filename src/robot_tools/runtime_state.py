"""Track which locked environment is installed in a shared runtime venv."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REAL_INSTALL_MARKER = ".robot-tools-real-install.json"


def real_install_marker(runtime_dir: str | Path) -> Path:
    return Path(runtime_dir) / ".venv" / REAL_INSTALL_MARKER


def real_lock_digest(runtime_dir: str | Path) -> str:
    lock_path = Path(runtime_dir) / "real" / "uv.lock"
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def record_real_install(runtime_dir: str | Path) -> Path:
    """Record a successfully verified real sync for later read-only checks."""
    runtime_dir = Path(runtime_dir).resolve()
    marker = real_install_marker(runtime_dir)
    if not marker.parent.is_dir():
        raise FileNotFoundError(f"runtime venv does not exist: {marker.parent}")

    payload = {
        "version": 1,
        "service_id": runtime_dir.name,
        "real_lock_sha256": real_lock_digest(runtime_dir),
    }
    temporary = marker.with_name(f"{marker.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(marker)
    return marker


def clear_real_install(runtime_dir: str | Path) -> None:
    """Clear real-mode state before restoring the lightweight environment."""
    real_install_marker(runtime_dir).unlink(missing_ok=True)


def validate_real_install(runtime_dir: str | Path) -> str | None:
    """Return a diagnostic when the venv is not marked for the current lock."""
    runtime_dir = Path(runtime_dir).resolve()
    marker = real_install_marker(runtime_dir)
    if not marker.is_file():
        return "real environment is not installed (run install)"

    try:
        payload = json.loads(marker.read_text())
    except (OSError, ValueError):
        return "real environment marker is invalid (rerun install)"
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return "real environment marker is invalid (rerun install)"
    if payload.get("service_id") != runtime_dir.name:
        return "real environment marker belongs to another service (rerun install)"

    try:
        expected_digest = real_lock_digest(runtime_dir)
    except OSError:
        return "real/uv.lock is missing"
    if payload.get("real_lock_sha256") != expected_digest:
        return "real environment does not match real/uv.lock (rerun install)"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("runtime_dir", type=Path)
    args = parser.parse_args()

    if args.command == "record":
        marker = record_real_install(args.runtime_dir)
        print(f"Recorded real runtime state: {marker}")


if __name__ == "__main__":
    main()
