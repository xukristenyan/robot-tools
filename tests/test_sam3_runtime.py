from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test job
    import tomli as tomllib


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtimes" / "sam3"


def _load_pyproject(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def test_fake_project_has_no_upstream_or_model_dependencies():
    project = _load_pyproject(RUNTIME_DIR / "pyproject.toml")

    assert project["project"]["dependencies"] == ["robot-tools[server]"]
    assert set(project["tool"]["uv"]["sources"]) == {"robot-tools"}


def test_real_project_is_locked_and_scoped_to_image_inference():
    project = _load_pyproject(RUNTIME_DIR / "real" / "pyproject.toml")
    dependencies = set(project["project"]["dependencies"])

    assert {
        "einops",
        "psutil",
        "pycocotools",
        "robot-tools[server]",
        "sam3",
        "setuptools>=61,<81",
        "torch==2.10.0",
        "torchvision==0.25.0",
    } == dependencies
    assert dependencies.isdisjoint(
        {
            "torchaudio==2.10.0",
            "hydra-core",
            "opencv-python-headless==4.11.0.86",
            "scipy",
            "scikit-image",
            "scikit-learn",
        }
    )

    sources = project["tool"]["uv"]["sources"]
    assert sources["sam3"] == {"path": "../upstream", "editable": True}
    assert sources["torch"] == {"index": "pytorch-cu128"}
    assert sources["torchvision"] == {"index": "pytorch-cu128"}
    assert project["tool"]["uv"]["required-version"] == ">=0.11.21"
    assert project["tool"]["uv"]["extra-build-dependencies"] == {"sam3": ["setuptools<81"]}

    lock_path = RUNTIME_DIR / "real" / "uv.lock"
    lock = lock_path.read_text()
    locked_package_names = {package["name"] for package in _load_pyproject(lock_path)["package"]}
    assert 'name = "torch"\nversion = "2.10.0+cu128"' in lock
    assert 'name = "torchvision"\nversion = "0.25.0+cu128"' in lock
    assert 'name = "sam3"\nsource = { editable = "../upstream" }' in lock
    for package in ("einops", "psutil", "pycocotools"):
        assert package in locked_package_names
    for package in ("torchaudio", "opencv-python", "hydra-core", "scipy", "scikit-image"):
        assert package not in locked_package_names


def test_install_script_uses_one_exact_real_sync():
    script = (RUNTIME_DIR / "install.sh").read_text()

    assert 'UV_PROJECT_ENVIRONMENT="$VENV_DIR"' in script
    assert 'uv sync --project "$REAL_PROJECT" --locked' in script
    assert "uv pip install" not in script
