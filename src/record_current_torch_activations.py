from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configs import REFERENCES
from encoding_utils import FourierEncoding
from load_data import load_video_frames
from nika import NikaBlock, tucker_construct


def _save(path: Path, tensor: torch.Tensor) -> None:
    np.save(path, tensor.detach().cpu().numpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="small")
    parser.add_argument("--frame-idx", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-frames", type=int, default=132)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vid = load_video_frames(args.frame_dir, args.device, max_frames=args.n_frames, dtype=torch.uint8, normalize=False)
    model = NikaBlock(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        **REFERENCES[args.config],
        out_channels=3,
        device=args.device,
    )
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        norm_t = torch.tensor([args.frame_idx / (vid.shape[0] - 1)], device=args.device, dtype=torch.float32)
        min_t = torch.max(torch.tensor(0.0, device=args.device), norm_t.min() - model.dT * model.operator_steps)
        max_t = torch.min(torch.tensor(1.0, device=args.device), norm_t.max() + model.dT * model.operator_steps)

        real_tucker = model.real_tucker(targets=(min_t, max_t), pad_to=model.B)
        grid_features = model.grid_features(model.B)

        complex_uh = model.complex_tucker.UH()
        complex_uw = model.complex_tucker.UW()
        complex_uc = model.complex_tucker.UC()
        complex_ut = model.complex_tucker.UT.get_range(targets=(min_t, max_t), pad_to=model.B)
        complex_core = torch.complex(model.complex_tucker.G_real, model.complex_tucker.G_imag)
        complex_construct = tucker_construct(complex_ut, complex_uc, complex_uh, complex_uw, complex_core)
        complex_grid = torch.complex(*model.complex_tucker.feature_grid(model.B).chunk(2, dim=1))
        complex_tucker = torch.fft.irfft2(complex_construct * complex_grid, norm="ortho").real

        response_input = torch.cat([real_tucker, grid_features, complex_tucker], dim=1)
        groupnorm = model.groupnorm(response_input)
        aggregated = groupnorm[model.operator_steps]
        operator_input = groupnorm.reshape(1, -1, groupnorm.shape[2], groupnorm.shape[3])

        operator_initial = model.flow_operator.operator_head(operator_input)
        time_emb = model.flow_operator.encoding(norm_t.view(-1, 1))
        modulation = model.flow_operator.t_modulator(time_emb)
        gamma, beta = modulation.chunk(2, dim=-1)
        gamma = gamma.view(-1, operator_initial.shape[1], 1, 1)
        beta = beta.view(-1, operator_initial.shape[1], 1, 1)
        operator_output = model.flow_operator.operator_tail(operator_initial * (1 + gamma) + beta)

        aggregated_with_operator = aggregated.unsqueeze(0) + operator_output
        output = model.upres(aggregated_with_operator)

    _save(out_dir / "norm_t.npy", norm_t)
    _save(out_dir / "real_tucker.npy", real_tucker)
    _save(out_dir / "grid_features.npy", grid_features)
    _save(out_dir / "complex_tucker_construct.npy", complex_construct)
    _save(out_dir / "complex_tucker_grid_real.npy", complex_grid.real)
    _save(out_dir / "complex_tucker_grid_imag.npy", complex_grid.imag)
    _save(out_dir / "complex_tucker.npy", complex_tucker)
    _save(out_dir / "response_input.npy", response_input)
    _save(out_dir / "groupnorm.npy", groupnorm)
    _save(out_dir / "operator_input.npy", operator_input)
    _save(out_dir / "operator_initial.npy", operator_initial)
    _save(out_dir / "operator_time_emb.npy", time_emb)
    _save(out_dir / "operator_gamma.npy", gamma)
    _save(out_dir / "operator_beta.npy", beta)
    _save(out_dir / "operator_output.npy", operator_output)
    _save(out_dir / "aggregated.npy", aggregated_with_operator)
    _save(out_dir / "output.npy", output)

    metadata = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "frame_idx": args.frame_idx,
        "n_frames": int(vid.shape[0]),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
