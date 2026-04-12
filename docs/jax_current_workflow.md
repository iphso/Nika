# Current Checkpoint JAX Workflow

This document covers the reproducible workflow for the current `HEAD` bunny checkpoint:

- exporting the Torch checkpoint to `.npz` for JAX
- recording Torch activations and running JAX parity checks
- rendering Torch and JAX videos and building a side-by-side comparison
- benchmarking Torch and JAX FPS

The commands below assume the current best checkpoint is:

`src/models/small-bunny-epoch200-psnr30.23-clean.torch`

## Environments

Use two environments:

- Torch / current repo code: the existing Docker Compose container
- JAX / CUDA 13: the local `.venv-jax13`

The Dockerfile now always installs a separate JAX venv:

```bash
docker build -f src/Dockerfile src
```

That installs JAX into `/opt/jax13-venv` with the same package family used in the local working setup. The current base image is still CUDA 11.8, so GPU JAX inside the container also requires moving that image to a CUDA 13 compatible runtime.

From the repo root:

```bash
docker compose run --rm --entrypoint bash backend
```

For JAX commands on the host, use:

```bash
env -u LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
```

## 1. Export Params For JAX

Run this in the Docker Compose container:

```bash
python3 /app/export_torch_checkpoint_npz.py \
  --checkpoint /app/models/small-bunny-epoch200-psnr30.23-clean.torch \
  --output /app/../tmp/current_head_params/small-bunny-epoch200-psnr30.23-clean.npz
```

That produces:

`tmp/current_head_params/small-bunny-epoch200-psnr30.23-clean.npz`

## 2. Record Torch Activations

Run this in the Docker Compose container:

```bash
python3 /app/record_current_torch_activations.py \
  --checkpoint /app/models/small-bunny-epoch200-psnr30.23-clean.torch \
  --frame-dir /app/static/benchmarks/bunny \
  --output-dir /app/../tmp/current_head_activations \
  --config small \
  --frame-idx 17 \
  --device cuda:0 \
  --n-frames 132
```

That produces:

`tmp/current_head_activations`

## 3. Run JAX Parity Check

Run this on the host:

```bash
env -u LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  .venv-jax13/bin/python -u src/jax_current_compare.py \
  --params tmp/current_head_params/small-bunny-epoch200-psnr30.23-clean.npz \
  --activations tmp/current_head_activations
```

This compares JAX activations against the Torch reference activations recorded above.

## 4. Render Torch Video

Run this in the Docker Compose container:

```bash
python3 /app/render_current_torch.py \
  --checkpoint /app/models/small-bunny-epoch200-psnr30.23-clean.torch \
  --frame-source /app/static/benchmarks/bunny \
  --frame-dir /app/../tmp/current_torch_render/frames \
  --mp4-output /app/../tmp/current_torch_render/videos/bunny-small-epoch200-psnr30.23.mp4 \
  --config small \
  --device cuda:0 \
  --batch-size 1 \
  --n-frames 132 \
  --fps 24
```

Notes:

- `batch-size=1` is the safest baseline for Torch rendering.
- The script writes PNG frames and then builds the MP4.

## 5. Render JAX Video

Run this on the host:

```bash
env -u LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  .venv-jax13/bin/python -u src/render_jax_current.py \
  --params tmp/current_head_params/small-bunny-epoch200-psnr30.23-clean.npz \
  --frame-dir tmp/current_jax_render/frames \
  --mp4-output tmp/current_jax_render/videos/bunny-small-epoch200-psnr30.23-jax.mp4 \
  --device-id 0 \
  --batch-size 32 \
  --n-frames 132 \
  --fps 24
```

Notes:

- `batch-size=32` is the current maintained fast path for JAX rendering.

## 6. Build Side-By-Side Comparison Video

Run this on the host:

```bash
mkdir -p tmp/current_render_compare/videos

ffmpeg -y \
  -i tmp/current_torch_render/videos/bunny-small-epoch200-psnr30.23.mp4 \
  -i tmp/current_jax_render/videos/bunny-small-epoch200-psnr30.23-jax.mp4 \
  -filter_complex "[0:v]drawtext=text='PyTorch':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6[v0];[1:v]drawtext=text='JAX':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6[v1];[v0][v1]hstack=inputs=2[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  tmp/current_render_compare/videos/pytorch_vs_jax_epoch200.mp4
```

That produces:

`tmp/current_render_compare/videos/pytorch_vs_jax_epoch200.mp4`

## 7. Benchmark Torch FPS

Run this in the Docker Compose container:

```bash
python3 /app/benchmark_current_torch.py \
  --checkpoint /app/models/small-bunny-epoch200-psnr30.23-clean.torch \
  --frame-source /app/static/benchmarks/bunny \
  --config small \
  --device cuda:0 \
  --batch-size 1 \
  --n-frames 132 \
  --warmup-iters 5 \
  --repeats 20
```

## 8. Benchmark JAX FPS

Run this on the host:

```bash
env -u LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  .venv-jax13/bin/python -u src/jax_current_benchmark.py \
  --params tmp/current_head_params/small-bunny-epoch200-psnr30.23-clean.npz \
  --device-id 0 \
  --batch-size 32 \
  --n-frames 132 \
  --warmup-iters 10 \
  --repeats 10
```

## Current Maintained Defaults

- Torch render baseline: `batch-size=1`
- Torch benchmark baseline: `batch-size=1`
- JAX render fast path: cached `norm_t` + `batch-size=32`
- JAX benchmark fast path: cached `norm_t` + `batch-size=32`

## Key Differences From The Torch Baseline

- The model math is unchanged. JAX still implements the same Tucker, complex grid, groupnorm, operator, and upres path.
- JAX precomputes frame-invariant work once in `prepare_runtime(...)`:
  - real Tucker spatial/channel basis
  - complex Tucker spatial/channel basis
  - sampled static feature grid
  - sampled complex feature grid
- JAX then batches multiple frame times with `vmap`, which is the main source of the current FPS win.
- The discarded experiments are not part of the maintained path:
  - no frame-index fast path
  - no full-sequence `scan` benchmark path
