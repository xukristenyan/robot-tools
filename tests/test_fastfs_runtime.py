from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test job
    import tomli as tomllib


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtimes" / "fastfs"


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def test_fake_project_has_no_upstream_or_model_dependencies():
    project = _load_toml(RUNTIME_DIR / "pyproject.toml")

    assert project["project"]["dependencies"] == ["robot-tools[server]"]
    assert set(project["tool"]["uv"]["sources"]) == {"robot-tools"}


def test_real_project_is_locked_and_scoped_to_pytorch_inference():
    project = _load_toml(RUNTIME_DIR / "real" / "pyproject.toml")
    dependencies = set(project["project"]["dependencies"])

    assert {
        "imageio",
        "omegaconf",
        "opencv-python-headless",
        "robot-tools[server]",
        "timm",
        "torch==2.10.0",
        "torchvision==0.25.0",
    } == dependencies
    assert dependencies.isdisjoint(
        {
            "einops",
            "open3d",
            "scikit-image",
            "scipy",
            "xformers==0.0.35",
        }
    )

    sources = project["tool"]["uv"]["sources"]
    assert sources["torch"] == {"index": "pytorch-cu128"}
    assert sources["torchvision"] == {"index": "pytorch-cu128"}
    assert project["tool"]["uv"]["required-version"] == ">=0.11.21"

    lock_path = RUNTIME_DIR / "real" / "uv.lock"
    lock = lock_path.read_text()
    locked_package_names = {package["name"] for package in _load_toml(lock_path)["package"]}
    assert 'name = "torch"\nversion = "2.10.0+cu128"' in lock
    assert 'name = "torchvision"\nversion = "0.25.0+cu128"' in lock
    for package in ("imageio", "omegaconf", "opencv-python-headless", "timm"):
        assert package in locked_package_names
    for package in ("einops", "open3d", "scikit-image", "scipy", "xformers"):
        assert package not in locked_package_names


def test_install_script_uses_one_exact_real_sync():
    script = (RUNTIME_DIR / "install.sh").read_text()

    assert 'UV_PROJECT_ENVIRONMENT="$VENV_DIR"' in script
    assert 'uv sync --project "$REAL_PROJECT" --locked' in script
    assert "uv pip install" not in script


def test_download_tool_is_pinned_and_isolated_from_runtime():
    script = (RUNTIME_DIR / "download.sh").read_text()

    assert 'GDOWN_VERSION="6.1.0"' in script
    assert 'uvx --from "gdown==$GDOWN_VERSION"' in script
    assert "VIRTUAL_ENV" not in script
