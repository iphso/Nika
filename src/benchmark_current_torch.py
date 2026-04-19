from __future__ import annotations

import argparse
import json
import re

import torch

from configs import REFERENCES
from load_data import load_video_frames
from nika import NikaBlock


def _strip_compile_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        stripped[re.sub(r"^_orig_mod\.", "", key)] = value
    return stripped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-source", required=True)
    parser.add_argument("--config", default="small")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-frames", type=int, default=132)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    args = parser.parse_args()

    if "cuda" not in args.device:
        raise ValueError("This benchmark uses CUDA timing; please use a CUDA device.")

    vid = load_video_frames(args.frame_source, args.device, max_frames=args.n_frames, dtype=torch.uint8, normalize=False)
    model = NikaBlock(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        **REFERENCES[args.config],
        out_channels=3,
        device=args.device,
    )
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(_strip_compile_prefixes(state))
    model.eval()
    model = torch.compile(model, mode=args.compile_mode)

    total_frames = int(vid.shape[0])
    batches = [
        torch.linspace(
            start / (total_frames - 1),
            (stop - 1) / (total_frames - 1),
            steps=stop - start,
            device=args.device,
            dtype=torch.float32,
        )
        for start in range(0, total_frames, args.batch_size)
        for stop in [min(start + args.batch_size, total_frames)]
    ]

    with torch.no_grad():
        for _ in range(args.warmup_iters):
            for norm_t in batches:
                _ = model(norm_t)
        torch.cuda.synchronize(args.device)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0

    with torch.no_grad():
        for repeat in range(args.repeats):
            torch.cuda.synchronize(args.device)
            starter.record()
            for norm_t in batches:
                _ = model(norm_t)
            ender.record()
            torch.cuda.synchronize(args.device)
            elapsed_ms = starter.elapsed_time(ender)
            total_ms += elapsed_ms
            print(json.dumps({"repeat": repeat + 1, "elapsed_s": elapsed_ms / 1000.0}))

    avg_s = (total_ms / args.repeats) / 1000.0
    fps = total_frames / avg_s
    print(json.dumps({
        "device": args.device,
        "n_frames": total_frames,
        "batch_size": args.batch_size,
        "avg_s_per_sequence": avg_s,
        "fps": fps,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
