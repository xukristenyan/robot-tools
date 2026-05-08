# Install GraspGen on R2D2

R2D2 has CUDA 13.x system-wide but PyTorch 2.1.0 requires CUDA 12.1. This recipe works around the version mismatch by installing CUDA 12.1 dev tools via conda and pointing the build to pip's nvidia runtime libraries.

## Prerequisites

- conda (miniconda/anaconda)
- `g++-12` installed (`sudo apt install gcc-12 g++-12`)
- NVIDIA GPU with compute capability 8.6 (adjust `TORCH_CUDA_ARCH_LIST` if different)

## Steps

```bash
# 1. Create conda env
conda create -n graspgen_base python=3.10 -y
conda activate graspgen_base

# 2. Install PyTorch + PyG
pip install torch==2.1.0 torchvision==0.16.0 torch-cluster torch-scatter \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 3. Install CUDA 12.1 dev tools (nvcc, headers, thrust)
conda install -c nvidia cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105 cuda-cccl=12.1.109 -y

# 4. Fix dangling libcudart symlink
#    conda creates a libcudart.so -> libcudart.so.12.1.105 symlink,
#    but only the 13.x runtime is installed. Point it to pip's 12.x lib instead.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
  $CONDA_PREFIX/lib/libcudart.so.12
ln -sf libcudart.so.12 $CONDA_PREFIX/lib/libcudart.so

# 5. Clone and install GraspGen
git clone https://github.com/NVlabs/GraspGen.git && cd GraspGen
pip install -e .

# 6. Build pointnet2_ops
#    Must set include paths to pip's nvidia headers and use gcc-12.
NVIDIA_PKG=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia \
CPLUS_INCLUDE_PATH=$NVIDIA_PKG/cuda_runtime/include:$NVIDIA_PKG/cublas/include:$NVIDIA_PKG/cusparse/include:$NVIDIA_PKG/cusolver/include \
C_INCLUDE_PATH=$NVIDIA_PKG/cuda_runtime/include:$NVIDIA_PKG/cublas/include:$NVIDIA_PKG/cusparse/include:$NVIDIA_PKG/cusolver/include \
CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12 \
CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.6" \
bash -c 'cd pointnet2_ops && rm -rf build *.egg-info && pip install --no-build-isolation .'

# 7. Verify
python tests/test_inference_installation.py
```

## Why this is needed

| Problem | Cause | Fix |
|---------|-------|-----|
| `CUDA version mismatch (13.x vs 12.1)` | System/conda CUDA is 13.x, torch compiled with 12.1 | Pin `cuda-nvcc=12.1.105` via conda |
| `cuda_runtime.h: No such file` | Headers live under `targets/x86_64-linux/include/` not `include/` | `cuda-cudart-dev=12.1.105` places them correctly |
| `thrust/complex.h: No such file` | Thrust headers not in standard include path | `cuda-cccl=12.1.109` provides matching thrust |
| `cannot find -lcudart` | Dangling symlink to nonexistent `libcudart.so.12.1.105` | Symlink to pip's `libcudart.so.12` |
| `unsupported GNU version! gcc > 12` | System g++ is 13.x, CUDA 12.1 max is gcc-12 | Use `gcc-12` / `g++-12` |
