import os
import argparse
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


from nika import NikaBlock
from soap import SOAP
from run_utils import add_common_args, resolve_device, setup_torch
from load_data import load_video_frames
from configs import REFERENCES


class GearThird(nn.Module):
    def __init__(self, channels, latent_dim, target_res, model_kwargs, device):
        super().__init__()
        self.target_res = target_res
        self.channels = channels
        self.device = device
        model_kwargs["device"] = device

        kwarg_list = self._inflate(target_res, latent_dim, channels, model_kwargs)
        self.blocks = nn.ModuleList([NikaBlock(**kwarg).to(device) for kwarg in kwarg_list])

        print(f"Total param count: {sum(p.numel() for p in self.parameters())}")

    @staticmethod
    def _inflate(target_res, latent_dim, out_channels, model_kwargs):
        kwarg_list = []
        full_res = [latent_dim, target_res[2], target_res[3], target_res[0]]
        ks = model_kwargs.pop("ks")
        tucker_scale = model_kwargs.pop("tucker_scale", 1/3)
        grid_scale = model_kwargs.pop("grid_scale", -2/3)
        current_res = full_res.copy()
        current_tucker_ranks = model_kwargs["real_tucker_ranks"]
        current_grid_ranks = model_kwargs["grid_ranks"]
        current_complex_tucker_ranks = model_kwargs["complex_tucker_ranks"]
        current_conv_hidden = model_kwargs["conv_hidden"]
        current_op_hdim = model_kwargs["op_hdim"]

        for i, k in enumerate(ks):
            kwarg_copy = model_kwargs.copy()
            kwarg_copy["k"] = k

            kwarg_copy["target_shape"] = current_res
            kwarg_copy["real_tucker_ranks"] = current_tucker_ranks
            kwarg_copy["grid_ranks"] = current_grid_ranks
            kwarg_copy["complex_tucker_ranks"] = current_complex_tucker_ranks
            kwarg_copy["conv_hidden"] = current_conv_hidden
            kwarg_copy["op_hdim"] = current_op_hdim

            if i == 0:
                kwarg_copy["out_channels"] = out_channels
            else:
                kwarg_copy["out_channels"] = full_res[0]

            if i == len(ks) - 1:
                kwarg_copy["accept_sidecar"] = False
            else:
                kwarg_copy["accept_sidecar"] = True
            kwarg_list.append(kwarg_copy)

            if i == len(ks) - 1:
                break

            # Kolmogorov-style rank deflation/inflation
            tucker_factor = (1 / k) ** (tucker_scale)
            grid_factor = (1 / k) ** (grid_scale)

            current_res = [current_res[0], current_res[1] // k, current_res[2] // k, current_res[3]]
            current_tucker_ranks = [
                current_tucker_ranks[0],
                int(current_tucker_ranks[1] * tucker_factor),
                int(current_tucker_ranks[2] * tucker_factor),
                int(current_tucker_ranks[3] * tucker_factor)
            ]

            current_complex_tucker_ranks = [
                current_complex_tucker_ranks[0],
                int(current_complex_tucker_ranks[1] * tucker_factor),
                int(current_complex_tucker_ranks[2] * tucker_factor),
                int(current_complex_tucker_ranks[3] * tucker_factor)
            ]
            # current_conv_hidden = int(current_conv_hidden * grid_factor)
            # current_op_hdim = int(current_op_hdim * grid_factor)
            current_grid_ranks = int(grid_factor * current_grid_ranks)
        return kwarg_list        

    def forward(self, x):
        accept_sidecar = False
        last_output = None
        for block in self.blocks[::-1]:
            if accept_sidecar:
                last_output = block(x, sidecar=last_output)
            else:
                last_output = block(x)
            accept_sidecar = True
        return last_output


def run():
    parser = argparse.ArgumentParser(description="Run GearThird experiment")
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    setup_torch()
    frames = load_video_frames(os.path.join(args.basedir, args.name), device=device)
    print(f"Loaded video frames with shape {frames.shape} on device {device}")
    config = REFERENCES[args.config]
    model = GearThird(channels=3, latent_dim=3, target_res=frames.shape, model_kwargs=config, device=device).to(device)

    batch_size = args.batch_size
    opt = SOAP(model.parameters(), lr=1e-3)

    batched_time = torch.arange(0, frames.shape[0], device=device)
    time_dataloader = torch.utils.data.DataLoader(batched_time, batch_size=batch_size, shuffle=True)

    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)

    for epoch in range(2000):
        t_start.record()
        opt.zero_grad()
        total_loss = 0.0
        for t in time_dataloader:
            gt = frames[t]
            norm_t = t / (frames.shape[0] - 1)
            test = model(norm_t)
            # kernel_gt = gt[..., ::model.ks[0], ::model.ks[0]]

            test_mse = F.mse_loss(test, gt)
            kernel_psnr = -10 * torch.log10(test_mse + 1e-8)
            # kernel_mse = F.mse_loss(kernel, kernel_gt)
            # kernel_psnr = -10 * torch.log10(kernel_mse + 1e-8)
            # pred_mse = F.mse_loss(pred, gt)
            # pred_psnr = -10 * torch.log10(pred_mse + 1e-8)
            # loss = -(kernel_psnr + pred_psnr)
            loss = -kernel_psnr * (batch_size / frames.shape[0])
            loss.backward()
            total_loss += kernel_psnr.item()
        opt.step()
        t_end.record()
        torch.cuda.synchronize()
        epoch_ms = t_start.elapsed_time(t_end)
        avg_psnr = total_loss / len(time_dataloader)
        print(f"Epoch {epoch} - PSNR: {avg_psnr:.4f} | {epoch_ms:.1f}ms ({epoch_ms / frames.shape[0]:.2f}ms/frame)")


if __name__ == "__main__":
    run()
