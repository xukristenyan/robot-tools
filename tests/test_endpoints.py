import pytest

from robot_tools.core.endpoints import get_endpoint, load_endpoints


def _write(tmp_path, text):
    p = tmp_path / "endpoints.yaml"
    p.write_text(text)
    return p


def test_load_endpoints(tmp_path):
    p = _write(
        tmp_path,
        ("endpoints:\n  fastfs:   { host: 127.0.0.1, port: 5556 }\n  graspgen: { host: gpu5090.local, port: 5557 }\n"),
    )
    eps = load_endpoints(p)
    assert eps["fastfs"].port == 5556
    assert eps["graspgen"].host == "gpu5090.local"
    assert eps["graspgen"].service_id == "graspgen"


def test_get_endpoint_unknown_name_lists_available(tmp_path):
    p = _write(tmp_path, "endpoints:\n  fastfs: { host: 127.0.0.1, port: 5556 }\n")
    with pytest.raises(KeyError, match="fastfs"):
        get_endpoint("sam3", p)


def test_file_without_endpoints_section_rejected(tmp_path):
    p = _write(tmp_path, "services: []\n")
    with pytest.raises(ValueError, match="endpoints"):
        load_endpoints(p)


def test_entry_missing_port_rejected(tmp_path):
    p = _write(tmp_path, "endpoints:\n  fastfs: { host: 127.0.0.1 }\n")
    with pytest.raises(ValueError, match="host"):
        load_endpoints(p)
