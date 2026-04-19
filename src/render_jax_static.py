from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

from jax_static_nika import forward, infer_spec, load_reference_params


def _to_uint8(image_chw: np.ndarray) -> np.ndarray:
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    return np.clip(np.rint(image_hwc * 255.0), 0, 255).astype(np.uint8)


def _make_mp4(frame_dir: Path, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="tmp/reference_params/small_bunny_static_model.npz")
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--mp4-output")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--n-frames", type=int)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    device = jax.devices()[args.device_id]
    params = jax.device_put(load_reference_params(args.params), device)
    spec = infer_spec(params)
    n_frames = spec.frames if args.n_frames is None else min(args.n_frames, spec.frames)

    @jax.jit
    def decode_one(frame_idx: jnp.ndarray) -> jnp.ndarray:
        return forward(params, frame_idx, spec)

    warmup = decode_one(jnp.asarray([0], dtype=jnp.int32))
    jax.block_until_ready(warmup)

    for frame_idx in range(n_frames):
        output = decode_one(jnp.asarray([frame_idx], dtype=jnp.int32))
        image = np.asarray(jax.device_get(output[0]), dtype=np.float32)
        iio.imwrite(frame_dir / f"frame_{frame_idx:04d}.png", _to_uint8(image))
        if frame_idx % 25 == 0 or frame_idx == n_frames - 1:
            print(f"rendered jax frame {frame_idx + 1}/{n_frames}")

    if args.mp4_output:
        _make_mp4(frame_dir, Path(args.mp4_output), args.fps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
