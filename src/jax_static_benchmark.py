from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from jax_static_nika import forward, infer_spec, load_reference_params


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        default="tmp/reference_params/small_bunny_static_model.npz",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--n-frames", type=int, default=132)
    args = parser.parse_args()

    device = jax.devices()[args.device_id]
    params = jax.device_put(load_reference_params(args.params), device)
    spec = infer_spec(params)
    n_frames = min(args.n_frames, spec.frames)

    @jax.jit
    def decode_one(frame_idx: jnp.ndarray) -> jnp.ndarray:
        return forward(params, frame_idx[None], spec)

    frame_indices = [jnp.asarray(i, dtype=jnp.int32) for i in range(n_frames)]

    for _ in range(args.warmup_iters):
        last = None
        for frame_idx in frame_indices:
            last = decode_one(frame_idx)
        jax.block_until_ready(last)

    total_s = 0.0
    for repeat in range(args.repeats):
        last = None
        start = time.perf_counter()
        for frame_idx in frame_indices:
            last = decode_one(frame_idx)
        jax.block_until_ready(last)
        elapsed = time.perf_counter() - start
        total_s += elapsed
        print(json.dumps({"repeat": repeat + 1, "elapsed_s": elapsed}))

    avg_s = total_s / args.repeats
    fps = n_frames / avg_s
    print(
        json.dumps(
            {
                "device": str(device),
                "n_frames": n_frames,
                "avg_s_per_sequence": avg_s,
                "fps": fps,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
