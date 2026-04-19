from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import imageio.v3 as iio
import torch


def _load_shape(frame_dir: str, n_frames: int | None) -> tuple[int, int, int]:
    paths = sorted(glob.glob(f"{frame_dir}/*.png"))
    if not paths:
        raise RuntimeError(f"no frames found in {frame_dir}")
    if n_frames is not None:
        paths = paths[:n_frames]
    first = iio.imread(paths[0], plugin="pillow")
    height, width = first.shape[:2]
    return len(paths), height, width


def _strip_compile_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = re.sub(r"^_orig_mod\.", "", key)
        stripped[new_key] = value
    return stripped


def _to_uint8(image_chw: torch.Tensor) -> torch.Tensor:
    image_hwc = image_chw.permute(1, 2, 0).contiguous()
    return torch.clamp(torch.round(image_hwc * 255.0), 0.0, 255.0).to(torch.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-source", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--config", default="small")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-frames", type=int)
    args = parser.parse_args()

    sys.path.insert(0, args.legacy_dir)

    torch.compile = lambda module, *a, **k: module

    from configs import REFERENCES
    from nika import NikaBlock

    total_frames, height, width = _load_shape(args.frame_source, args.n_frames)
    out_dir = Path(args.frame_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = NikaBlock(
        target_shape=[4, height, width, total_frames],
        k=4,
        **REFERENCES[args.config],
        out_channels=3,
        device=args.device,
    )
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(_strip_compile_prefixes(state_dict))
    model.eval()

    with torch.no_grad():
        for start in range(0, total_frames, args.batch_size):
            end = min(start + args.batch_size, total_frames)
            t_batch = torch.arange(start, end, device=args.device, dtype=torch.int64)
            predictions = model(t_batch).clamp(0.0, 1.0)
            for offset, image in enumerate(predictions):
                iio.imwrite(
                    out_dir / f"frame_{start + offset:04d}.png",
                    _to_uint8(image).cpu().numpy(),
                )
            print(f"rendered pytorch frames {start + 1}-{end}/{total_frames}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
