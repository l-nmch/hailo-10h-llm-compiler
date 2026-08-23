# Toolchain containers

Everything in this project runs inside Docker. Nothing is ever installed on
the host machine (except Docker itself, the Hailo PCIe driver on the device
host, and the HailoRT/hailo-ollama user-space packages there — see
[../docs/device-setup.md](../docs/device-setup.md)).

Three images are provided:

| Image | Dockerfile | Purpose |
|---|---|---|
| `dfc-nvidia` | [Dockerfile.nvidia](Dockerfile.nvidia) | Full compile pipeline on an NVIDIA GPU (CUDA) |
| `dfc-amd` | [Dockerfile.amd](Dockerfile.amd) | Same pipeline on an AMD GPU (ROCm) |
| `dfc-jupyter` | [jupyter.Dockerfile](jupyter.Dockerfile) | Jupyter layer on top of either image, for interactive experimentation |

## The Dataflow Compiler wheel — bring your own

**This repository does not redistribute the Hailo Dataflow Compiler.** The
DFC is proprietary software: it requires a (free) Hailo account and an
explicit license acceptance, and its license does not permit redistribution.
The device firmware is likewise proprietary.

To obtain it:

1. Create an account on the [Hailo Developer Zone](https://hailo.ai/developer-zone/)
   (free).
2. Download the **Hailo Software Suite** package for your platform.
3. From the package, extract the compiler wheel, named like:
   ```
   hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl
   ```
4. Copy that wheel next to the Dockerfile you intend to build:
   ```bash
   cp hailo_dataflow_compiler-*.whl docker/
   ```
   (The wheel is git-ignored; it must never be committed.)

All Dockerfiles expect the wheel at `/tmp/hailo_dataflow_compiler-*.whl`
in their build context and fail with a clear message if it is missing.

## Building

### NVIDIA (CUDA)

```bash
cp hailo_dataflow_compiler-*.whl docker/
docker build -f docker/Dockerfile.nvidia -t dfc-nvidia:5.3.0 docker/
```

### AMD (ROCm)

The AMD image builds on a community ROCm PyTorch base image so that the
DFC's optimization engine can run on AMD GPUs (the stock DFC optimizer is
TensorFlow/CUDA-oriented; this project switches it to the PyTorch engine —
see [../docs/findings/sdk-behavior-notes.md](../docs/findings/sdk-behavior-notes.md)).
Tested against `mixa3607/pytorch-gfx906:v2.11.0-rocm-7.2.1`; any ROCm
PyTorch base close to your kernel/GPU should work the same way.

```bash
cp hailo_dataflow_compiler-*.whl docker/
docker build -f docker/Dockerfile.amd -t dfc-amd:5.3.0 docker/
```

### Jupyter (on top of either)

```bash
docker build -f docker/jupyter.Dockerfile \
    --build-arg BASE_IMAGE=dfc-nvidia:5.3.0 \
    -t dfc-jupyter:nvidia docker/

# or: --build-arg BASE_IMAGE=dfc-amd:5.3.0 -t dfc-jupyter:amd
```

## Running

### NVIDIA

```bash
docker run --rm -it --gpus all \
    --memory=24g \
    -v "$PWD/workdir":/workdir \
    dfc-nvidia:5.3.0 bash
```

### AMD

AMD GPU passthrough needs more flags than NVIDIA (`/dev/kfd`, DRM devices,
render/video groups):

```bash
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add $(getent group render | cut -d: -f3) \
    --group-add $(getent group video  | cut -d: -f3) \
    --security-opt seccomp=unconfined --ipc=host \
    --memory=24g \
    -v "$PWD/workdir":/workdir \
    dfc-amd:5.3.0 bash
```

### Environment variables — do not skip these

Both images bake in two critical settings; they are repeated here because
they are easy to lose when overriding the environment:

| Variable | Why |
|---|---|
| `NPY_PROMOTION_STATE=legacy` | The quantization stage mixes float32/float64 arrays; under NumPy ≥ 2 semantics the result differs and `create_hw_params()` fails. Legacy promotion restores the expected behavior. |
| `CUDA_VISIBLE_DEVICES=<gpu id>` | Must be **set explicitly at run time**. If unset, the DFC sets it to `"99"` internally and no GPU is visible to PyTorch/TensorFlow. On AMD (ROCm), the variable name is still `CUDA_VISIBLE_DEVICES` — PyTorch's ROCm build honors it. |

Example: `docker run ... -e CUDA_VISIBLE_DEVICES=0 ...`

### Memory limits

The DFC optimization step can request large amounts of RAM. On hosts with
~32 GB, cap the container (`--memory=24g`) so an unlucky graph cannot take
the machine down. See
[../docs/device-setup.md#memory-limits](../docs/device-setup.md#memory-limits).

## Verifying the toolchain

Inside the container:

```bash
python -c "import hailo_dataflow_compiler" && echo "DFC OK"
python -c "import torch; print(torch.cuda.is_available())"   # True on both platforms
echo "$NPY_PROMOTION_STATE"                                   # legacy
```
