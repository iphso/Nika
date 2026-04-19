from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import imageio.v3 as iio
import torch

from configs import REFERENCES
from load_data import load_video_frames
from nika import NikaBlock


def _strip_compile_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        stripped[re.sub(r"^_orig_mod\.", "", key)] = value
    return stripped


def _to_uint8(image_chw: torch.Tensor) -> torch.Tensor:
    image_hwc = image_chw.permute(1, 2, 0).contiguous()
    return torch.clamp(torch.round(image_hwc * 255.0), 0.0, 255.0).to(torch.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-source", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--mp4-output", required=True)
    parser.add_argument("--config", default="small")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-frames", type=int, default=132)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    mp4_output = Path(args.mp4_output)
    mp4_output.parent.mkdir(parents=True, exist_ok=True)

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
    with torch.no_grad():
        for start in range(0, total_frames, args.batch_size):
            end = min(start + args.batch_size, total_frames)
            norm_t = torch.linspace(
                start / (total_frames - 1),
                (end - 1) / (total_frames - 1),
                steps=end - start,
                device=args.device,
                dtype=torch.float32,
            )
            predictions = model(norm_t).clamp(0.0, 1.0)
            for offset, image in enumerate(predictions):
                iio.imwrite(frame_dir / f"frame_{start + offset:04d}.png", _to_uint8(image).cpu().numpy())
            if end % 25 == 0 or end == total_frames:
                print(f"rendered frame {end}/{total_frames}")

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
            str(mp4_output),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
