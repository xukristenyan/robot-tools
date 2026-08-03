from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test job
    import tomli as tomllib


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtimes" / "graspgenx"


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def test_fake_project_has_no_upstream_or_model_dependencies():
    project = _load_toml(RUNTIME_DIR / "pyproject.toml")

    assert project["project"]["dependencies"] == ["robot-tools[server]"]
    assert set(project["tool"]["uv"]["sources"]) == {"robot-tools"}
    assert "override-dependencies" not in project["tool"]["uv"]


def test_real_project_owns_only_http_inference_dependencies():
    project = _load_toml(RUNTIME_DIR / "real" / "pyproject.toml")
    dependencies = set(project["project"]["dependencies"])

    assert {
        "diffusers==0.11.1",
        "huggingface-hub==0.25.2",
        "numpy==1.26.4",
        "omegaconf",
        "pyrender==0.1.45",
        "robot-tools[server]",
        "scipy",
        "timm==1.0.15",
        "torch==2.10.0",
        "torchvision==0.25.0",
        "trimesh==4.5.3",
    }.issubset(dependencies)
    assert dependencies.isdisjoint(
        {
            "hydra-core",
            "matplotlib",
            "pyzmq",
            "scene-synthesizer",
            "sharedarray",
            "tensorboard",
            "tensordict",
            "torch-geometric",
            "transformers",
            "viser",
        }
    )

    sources = project["tool"]["uv"]["sources"]
    assert set(sources) == {"robot-tools", "torch", "torchvision"}
    assert sources["torch"] == {"index": "pytorch-cu128"}
    assert sources["torchvision"] == {"index": "pytorch-cu128"}
    assert project["tool"]["uv"]["required-version"] == ">=0.11.21"


def test_real_lock_contains_exact_gpu_packages_without_training_stack():
    lock_path = RUNTIME_DIR / "real" / "uv.lock"
    lock = lock_path.read_text()
    names = {package["name"] for package in _load_toml(lock_path)["package"]}

    assert 'name = "torch"\nversion = "2.10.0+cu128"' in lock
    assert 'name = "torchvision"\nversion = "0.25.0+cu128"' in lock
    assert {"diffusers", "pyrender", "timm", "trimesh"}.issubset(names)
    assert {
        "hydra-core",
        "matplotlib",
        "pyzmq",
        "scene-synthesizer",
        "sharedarray",
        "tensorboard",
        "tensordict",
        "torch-geometric",
        "transformers",
        "viser",
    }.isdisjoint(names)


def test_install_is_locked_and_disables_import_time_downloads():
    script = (RUNTIME_DIR / "install.sh").read_text()
    download_patch = (RUNTIME_DIR / "compat" / "no-auto-download.patch").read_text()
    zmq_patch = (RUNTIME_DIR / "compat" / "lazy-zmq.patch").read_text()

    assert 'UV_PROJECT_ENVIRONMENT="$VENV_DIR"' in script
    assert 'uv sync --project "$REAL_PROJECT" --locked' in script
    assert "uv pip install" not in script
    assert "-    ensure_gripper_descriptions()" in download_patch
    assert "-    ensure_checkpoints()" in download_patch
    assert "def __getattr__(name)" in zmq_patch
    assert "from graspgenx.serving.zmq_client" in zmq_patch
