from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test job
    import tomli as tomllib


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtimes" / "graspgen"


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def test_fake_project_has_no_upstream_or_model_dependencies():
    project = _load_toml(RUNTIME_DIR / "pyproject.toml")

    assert project["project"]["dependencies"] == ["robot-tools[server]"]
    assert set(project["tool"]["uv"]["sources"]) == {"robot-tools"}
    assert "override-dependencies" not in project["tool"]["uv"]


def test_real_project_owns_inference_and_cuda_dependencies():
    project = _load_toml(RUNTIME_DIR / "real" / "pyproject.toml")
    dependencies = set(project["project"]["dependencies"])

    assert {
        "diffusers==0.11.1",
        "huggingface-hub==0.25.2",
        "numpy==1.26.4",
        "pointnet2-ops==3.0.0",
        "robot-tools[server]",
        "spconv-cu126==2.3.8",
        "timm==1.0.15",
        "torch==2.10.0",
        "torch-scatter==2.1.2+pt210cu128",
        "torchvision==0.25.0",
    }.issubset(dependencies)
    assert dependencies.isdisjoint(
        {
            "open3d",
            "pyrender",
            "tensorboardx",
            "wandb",
            "usd-core",
        }
    )

    sources = project["tool"]["uv"]["sources"]
    assert sources["pointnet2-ops"] == {"path": "../upstream/pointnet2_ops"}
    assert sources["torch"] == {"index": "pytorch-cu128"}
    assert sources["torchvision"] == {"index": "pytorch-cu128"}
    assert project["tool"]["uv"]["required-version"] == ">=0.11.21"
    assert project["tool"]["uv"]["extra-build-dependencies"]["pointnet2-ops"][0] == {
        "requirement": "torch",
        "match-runtime": True,
    }


def test_real_lock_contains_exact_gpu_packages():
    lock_path = RUNTIME_DIR / "real" / "uv.lock"
    lock = lock_path.read_text()
    names = {package["name"] for package in _load_toml(lock_path)["package"]}

    assert 'name = "torch"\nversion = "2.10.0+cu128"' in lock
    assert 'name = "torchvision"\nversion = "0.25.0+cu128"' in lock
    assert 'name = "torch-scatter"\nversion = "2.1.2+pt210cu128"' in lock
    assert {"pointnet2-ops", "spconv-cu126", "diffusers", "timm"}.issubset(names)
    assert {"open3d", "pyrender", "tensorboardx", "wandb", "usd-core"}.isdisjoint(names)


def test_install_script_uses_one_exact_real_sync():
    script = (RUNTIME_DIR / "install.sh").read_text()

    assert 'UV_PROJECT_ENVIRONMENT="$VENV_DIR"' in script
    assert 'uv sync --project "$REAL_PROJECT" --locked' in script
    assert "uv pip install" not in script


def test_download_defaults_to_runtime_storage_variable():
    script = (RUNTIME_DIR / "download.sh").read_text()

    assert 'DEST="${DEST:-${GRASPGEN_MODELS_DIR:-$HERE/GraspGenModels}}"' in script
