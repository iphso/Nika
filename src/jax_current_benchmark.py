from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from jax_current_nika import forward, infer_spec, load_params, prepare_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--n-frames", type=int, default=132)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = jax.devices()[args.device_id]
    params = jax.device_put(load_params(args.params), device)
    spec = infer_spec(params)
    runtime = prepare_runtime(params, spec)
    n_frames = min(args.n_frames, spec.frames)

    @jax.jit
    def decode_one(norm_t: jnp.ndarray) -> jnp.ndarray:
        return forward(runtime, norm_t, spec)

    @jax.jit
    def decode_batch(norm_t_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda nt: forward(runtime, nt[None], spec)[0])(norm_t_batch)

    batches = [
        jnp.asarray([[i / (n_frames - 1)] for i in range(start, min(start + args.batch_size, n_frames))], dtype=jnp.float32)
        for start in range(0, n_frames, args.batch_size)
    ]

    for _ in range(args.warmup_iters):
        last = None
        if args.batch_size == 1:
            for batch in batches:
                last = decode_one(batch[0])
        else:
            for batch in batches:
                last = decode_batch(batch)
        jax.block_until_ready(last)

    total_s = 0.0
    for repeat in range(args.repeats):
        last = None
        start = time.perf_counter()
        if args.batch_size == 1:
            for batch in batches:
                last = decode_one(batch[0])
        else:
            for batch in batches:
                last = decode_batch(batch)
        jax.block_until_ready(last)
        elapsed = time.perf_counter() - start
        total_s += elapsed
        print(json.dumps({"repeat": repeat + 1, "elapsed_s": elapsed}))

    avg_s = total_s / args.repeats
    fps = n_frames / avg_s
    print(json.dumps({
        "device": str(device),
        "n_frames": n_frames,
        "batch_size": args.batch_size,
        "avg_s_per_sequence": avg_s,
        "fps": fps,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
