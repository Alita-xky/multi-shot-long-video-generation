# ViPE deployment stopped

ViPE was not run. No point cloud, Sim(3), or essential-matrix outputs were produced.
No fallback to CameraCtrl nominal poses, guessed HFOV, or Depth Anything was used.

## What the official installer requires

- Package: `nvidia-vipe` (https://github.com/nv-tlabs/vipe), current release 1.2.0
- `setup.py` imports `torch.utils.cpp_extension.CUDAExtension` and asserts `torch.version.cuda is not None` with the message `Pytorch CUDA is required for this installation.`
- Documented environment is conda `envs/cu128.yml` plus a CUDA toolkit that provides `nvcc`.
- Inference (`vipe infer VIDEO.mp4`) then needs an NVIDIA GPU to execute those CUDA kernels and to download the pose/depth models.

## What this machine has

- Architecture: aarch64
- Accelerators: Ascend NPU only (`/dev/davinci0`–`/dev/davinci7`). No `/dev/nvidia*`.
- `nvidia-smi`: not available
- System `nvcc`: not on PATH. No `/usr/local/cuda`.
- `h3` env: `torch 2.10.0+cpu`, `torch.version.cuda is None`, `torch.cuda.is_available() is False`
- `vllm` env: same CPU torch build, `torch.version.cuda is None`. A bundled `nvcc` exists under that env's `nvidia/cu13` Python package, but there is no NVIDIA device to run a CUDA extension.
- `import vipe`: not installed

Installing `nvidia-vipe` here fails at the CUDA extension build before any pose or depth can be estimated.
