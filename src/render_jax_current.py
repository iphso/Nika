from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

from jax_current_nika import forward, infer_spec, load_params, prepare_runtime


def _to_uint8(image_chw: np.ndarray) -> np.ndarray:
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    return np.clip(np.rint(image_hwc * 255.0), 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--mp4-output", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=132)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    device = jax.devices()[args.device_id]
    params = jax.device_put(load_params(args.params), device)
    spec = infer_spec(params)
    runtime = prepare_runtime(params, spec)
    n_frames = min(args.n_frames, spec.frames)

    @jax.jit
    def decode_batch(norm_t_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda nt: forward(runtime, nt[None], spec)[0])(norm_t_batch)

    warmup_count = min(args.batch_size, n_frames)
    warmup = decode_batch(
        jnp.asarray([[i / (n_frames - 1)] for i in range(warmup_count)], dtype=jnp.float32)
    )
    jax.block_until_ready(warmup)

    rendered = 0
    for start in range(0, n_frames, args.batch_size):
        stop = min(start + args.batch_size, n_frames)
        norm_t_batch = jnp.asarray([[i / (n_frames - 1)] for i in range(start, stop)], dtype=jnp.float32)
        outputs = np.asarray(jax.device_get(decode_batch(norm_t_batch)), dtype=np.float32)
        for offset, image in enumerate(outputs):
            idx = start + offset
            iio.imwrite(frame_dir / f"frame_{idx:04d}.png", _to_uint8(np.clip(image, 0.0, 1.0)))
        rendered = stop
        if rendered % 25 == 0 or rendered == n_frames:
            print(f"rendered frame {rendered}/{n_frames}")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            args.mp4_output,
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
