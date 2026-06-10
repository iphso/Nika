import argparse
import re
import os
import time
import math
import glob
from functools import partial

import torch
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F

import numpy as np
import imageio.v3 as iio
from PIL import Image

from soap import SOAP
import random
from load_data import load_video_frames
from encoding_utils import FourierEncoding
from configs import REFERENCES


class TuckerFactor(nn.Module):
    def __init__(self, target_dim, rank, is_complex=False, base_mag=1e-2, device='cuda'):
        """
        Have to split into chunks because there's some weird bug in PyTorch
        where if the dim is over like 520 or something everything just breaks.

        Perhaps someone can figure that out later, but the workaround seems easier atm.
        """
        super().__init__()
        self.target_dim = target_dim
        self.rank = rank
        self.is_complex = is_complex
        self.device = device

        # Simplified: no chunking needed for current usage (dims <= 500)
        if self.is_complex:
            self.U_real = nn.Parameter(torch.randn(target_dim, rank, device=device) * base_mag)
            self.U_imag = nn.Parameter(torch.zeros(target_dim, rank, device=device))
        else:
            self.U = nn.Parameter(torch.randn(target_dim, rank, device=device) * base_mag)

    def forward(self):
        def _col_norm(M, eps=1e-8):
            if torch.is_complex(M):
                norms_sq = (M.real**2 + M.imag**2).sum(dim=0, keepdim=True)
                norms = torch.sqrt(norms_sq + eps)
            else:
                norms = M.norm(dim=0, keepdim=True) + eps
            return M / norms
        if self.is_complex:
            return _col_norm(torch.complex(self.U_real, self.U_imag))
        return _col_norm(self.U)

    def get(self, target):
        U = self.forward()
        target = torch.as_tensor(target, device=U.device, dtype=torch.float32).view(-1)
        # clamp to [0,1]
        target = target.clamp(0.0, 1.0)

        # continuous positions in [0, D-1]
        D = self.target_dim
        pos = target * (D - 1)
        idx0 = pos.floor().long()
        idx1 = (idx0 + 1).clamp(max=D - 1)
        w = (pos - idx0.float()).unsqueeze(-1)  # [B,1]

        if torch.is_complex(U):
            real = U.real
            imag = U.imag
            s0_r = real[idx0]
            s1_r = real[idx1]
            s0_i = imag[idx0]
            s1_i = imag[idx1]
            out_r = (1.0 - w) * s0_r + w * s1_r
            out_i = (1.0 - w) * s0_i + w * s1_i
            return torch.complex(out_r, out_i)

        # real case
        s0 = U[idx0]
        s1 = U[idx1]
        out = (1.0 - w) * s0 + w * s1
        return out


class RealTucker(nn.Module):
    def __init__(self, target_shape, ranks, device='cuda'):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.rC, self.rH, self.rW, self.rT = ranks

        self.UH = TuckerFactor(self.H, self.rH, is_complex=False, device=device)
        self.UW = TuckerFactor(self.W, self.rW, is_complex=False, device=device)
        self.UC = TuckerFactor(self.C, self.rC, is_complex=False, device=device)
        self.UT = TuckerFactor(self.T, self.rT, is_complex=False, device=device)

        self.G = nn.Parameter(torch.randn(self.rT, self.rC, self.rH, self.rW, device=device) * 1e-2)

    def forward(self, t):
        UT = self.UT.get(t)
        UC = self.UC()
        UH = self.UH()
        UW = self.UW()
        return tucker_construct(UT, UC, UH, UW, self.G).contiguous()


class ComplexTucker(RealTucker):

    def __init__(self, target_shape, ranks, grid_channels, device='cuda'):
        """
        `grid_channels`: optional int to specify number of channels for the complex feature grid.
        If None, defaults to `self.C * 2` (legacy behavior).
        """
        self._has_tucker = bool(ranks)
        self._has_grid = bool(grid_channels)

        if self._has_tucker:
            super().__init__(target_shape, ranks, device=device)
            half_W = (self.W // 2) + 1
            self.UH = TuckerFactor(self.H, self.rH, is_complex=True, device=device)
            self.UW = TuckerFactor(half_W, self.rW, is_complex=True, device=device)
            self.UC = TuckerFactor(self.C, self.rC, is_complex=True, device=device)
            self.UT = TuckerFactor(self.T, self.rT, is_complex=True, device=device)

            self.G = None  # override parent
            self.G_real = nn.Parameter(torch.randn(self.rT, self.rC, self.rH, self.rW, device=device) * 1e-2)
            self.G_imag = nn.Parameter(torch.zeros(self.rT, self.rC, self.rH, self.rW, device=device))
        else:
            # no Tucker; initialize Module base so we can attach submodules
            nn.Module.__init__(self)
            # populate shape fields so grid-only mode can construct FeatureGrid
            C, H, W, T = target_shape
            self.C, self.H, self.W, self.T = C, H, W, T
            half_W = (W // 2) + 1

        if self._has_grid:
            # create FeatureGrid as a proper submodule
            self.feature_grid = FeatureGrid([self.C * 2, self.H, half_W, self.T], grid_res=grid_channels * 2, device=device)

    def forward(self, t, zero_tucker=False, zero_grid=False):
        # if both disabled, return zeros
        if zero_tucker and zero_grid:
            return torch.zeros((t.shape[0], self.C, self.H, self.W), device=t.device)

        construct = None
        if self._has_tucker and not zero_tucker:
            UH = self.UH()
            UW = self.UW()
            UC = self.UC()
            UT = self.UT.get(t)
            G = torch.complex(self.G_real, self.G_imag)
            construct = tucker_construct(UT, UC, UH, UW, G)

        if self._has_grid and not zero_grid:
            grid = self.feature_grid(t)
            complex_grid = torch.complex(*grid.chunk(2, dim=1))
            if construct is not None:
                construct = construct * complex_grid
            else:
                construct = complex_grid

        if construct is None:
            return torch.zeros((t.shape[0], self.C, self.H, self.W), device=t.device)

        real_tucker = torch.fft.irfft2(construct, norm='ortho').real
        return real_tucker.contiguous()


def grid_sample_base(H, W, device):
    y_lin = torch.arange(0, H, device=device)
    x_lin = torch.arange(0, W, device=device)
    y_norm = 2.0 * (y_lin / (H - 1)) - 1.0
    x_norm = 2.0 * (x_lin / (W - 1)) - 1.0
    y, x = torch.meshgrid(y_norm, x_norm, indexing='ij')  # [H, W]
    return torch.stack((x, y), dim=-1)  # [H, W, 2]


class FeatureGrid(nn.Module):
    def __init__(self, target_shape, grid_res, zero_init=False, device="cuda"):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        # `grid_res` is channels-only (integer): number of grid channels.
        # Spatial (H,W) uses full target (already downsampled by k upstream), temporal set to 1.
        self.grid_c = int(grid_res)
        self.grid_h = self.H
        self.grid_w = self.W
        self.grid_t = 1

        self.grid = nn.Parameter(torch.randn(self.grid_c, self.grid_h, self.grid_w, self.grid_t, device=device) * 1e-2)
        if self.grid_c != self.C:
            self.channel_proj = nn.Linear(self.grid_c, self.C, bias=True).to(device)
            nn.init.normal_(self.channel_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.channel_proj.bias)
        self.register_buffer(
            "_xy_base",
            grid_sample_base(self.H, self.W, device=device),
            persistent=False
        )

        self._grid_5d_view = None
    
    def _5d_grid(self):
        return self.grid.permute(0, 3, 1, 2).unsqueeze(0)

    def forward(self, t):
        B = t.shape[0]
        grid_spatial = self.grid.squeeze(-1)  # [grid_c, H, W]
        result = grid_spatial.unsqueeze(0).expand(B, -1, -1, -1)  # [B, grid_c, H, W]

        if hasattr(self, 'channel_proj'):
            result = self.channel_proj(result.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return result.contiguous()
        return result


def tucker_construct(UT, UC, UH, UW, G):
    return torch.einsum('ijkl,ti,cj,hk,wl->tchw', G, UT, UC, UH, UW)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, device='cuda'):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim).to(device)
        self.norm = nn.LayerNorm(dim, eps=1e-6).to(device)
        self.pw_conv1 = nn.Linear(dim, 4 * dim).to(device)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim).to(device)

        nn.init.zeros_(self.pw_conv2.weight)
        nn.init.zeros_(self.pw_conv2.bias)

    def forward(self, x):
        identity = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x.permute(0, 3, 1, 2)
        return identity + x


class BasicUpres(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, k, encoding_len=64, device='cuda'):
        super().__init__()
        self.k = k

        self.project_in = nn.Conv2d(in_channels, hidden, kernel_size=1).to(device)
        self.convnext = ConvNeXtBlock(hidden, device=device)
        self.project_out = nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=1).to(device)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=k)

        for m in [self.project_in, self.project_out]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.convnext(x)
        x = self.project_out(x)
        return self.pixel_shuffle(x)


class ConvOperator(nn.Module):
    def __init__(self, in_channels, out_channels, h_dim, encoding_len=128, device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.operator_head = nn.Sequential(
            nn.Conv2d(in_channels, h_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1, groups=h_dim),
            nn.GELU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=1),
        ).to(device)

        self.operator_tail = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1, groups=h_dim),
            nn.GELU(),
            nn.Conv2d(h_dim, out_channels, kernel_size=1),
        ).to(device)

        self.encoding = FourierEncoding(
            target_dim=encoding_len,
            max_freq=64,
            freq_init="log",
            device=device
        )

        self.t_modulator = nn.Sequential(
            nn.Linear(encoding_len, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 2 * h_dim),
        ).to(device)

        nn.init.zeros_(self.operator_tail[-1].weight)
        nn.init.zeros_(self.operator_tail[-1].bias)
        nn.init.zeros_(self.t_modulator[-1].weight)
        nn.init.zeros_(self.t_modulator[-1].bias)

    def forward(self, x, t):
        initial = self.operator_head(x)
        time_emb = self.encoding(t)
        modulation = self.t_modulator(time_emb)
        gamma, beta = modulation.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.operator_head[-1].out_channels, 1, 1)
        beta = beta.view(-1, self.operator_head[-1].out_channels, 1, 1)
        modulated = initial * (1 + gamma) + beta
        conv_x = self.operator_tail(modulated)
        return conv_x


class NikaBlock(nn.Module):
    def __init__(
        self,
        target_shape,
        k,
        real_tucker_ranks,
        complex_tucker_ranks,
        grid_ranks,
        conv_hidden,
        out_channels,
        operator_steps,
        op_hdim,
        device,
        complex_grid_ranks=None,
        use_compile=False
    ):
        super().__init__()
        self.C, self.H, self.W, T = target_shape
        self.orig_T = T
        self.T = T + (operator_steps * 2)  # Virtual padding
        self.H = int(self.H // k); self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]  # extra time steps for operator predictions
        self.dT = 1.0 / (self.T - 1)
        self.n_heads = 0
        # keep deterministic head order for concatenation
        self._head_order = []
        if real_tucker_ranks:
            self.real_tucker = RealTucker(
                target_shape=self.internal_shape,
                ranks=real_tucker_ranks,
                device=device,
            )
            self.n_heads += 1
            self._head_order.append('real')

        if grid_ranks:
            self.grid_features = FeatureGrid(
                target_shape=self.internal_shape,
                grid_res=grid_ranks,
                device=device,
            )
            self.n_heads += 1
            self._head_order.append('grid')

        complex_grid_ranks = complex_grid_ranks if complex_grid_ranks is not None else grid_ranks

        if complex_tucker_ranks or complex_grid_ranks:
            self.complex_tucker = ComplexTucker(
                target_shape=self.internal_shape,
                ranks=complex_tucker_ranks,
                grid_channels=complex_grid_ranks,
                device=device,
            )
            self.n_heads += 1
            self._head_order.append('complex')

        self.groupnorm = nn.GroupNorm(num_groups=self.n_heads, num_channels=self.n_heads * self.C).to(device)
        self.operator_steps = operator_steps
        self.register_buffer(
            "_zero_base",
            torch.zeros(1, self.C, self.H, self.W, device=device, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_operator_offsets",
            (torch.arange(-self.operator_steps, self.operator_steps + 1, device=device, dtype=torch.float32) * self.dT).unsqueeze(0),
            persistent=False,
        )

        self.forward_operators = nn.ModuleList()
        self.backward_operators = nn.ModuleList()
        for _ in range(self.operator_steps):
            fwd = ConvOperator(
                in_channels = 2 * self.n_heads * self.C,
                out_channels = self.n_heads * self.C,
                h_dim = op_hdim,
                device = device,
            )
            bwd = ConvOperator(
                in_channels = 2 * self.n_heads * self.C,
                out_channels = self.n_heads * self.C,
                h_dim = op_hdim,
                device = device,
            )
            self.forward_operators.append(fwd)
            self.backward_operators.append(bwd)

        self.upres = BasicUpres(
            in_channels = self.n_heads * self.C,
            out_channels = out_channels,
            hidden = conv_hidden,
            k = k,    
            device = device,
        )

        self.log_stats()

    def log_stats(self):
        total_params = 0
        print(f"NikaBlock parameters:")
        if hasattr(self, 'real_tucker'):
            real_tucker_params = sum(p.numel() for p in self.real_tucker.parameters())
            print(f"  Real Tucker:     {real_tucker_params / 1e6:.3f}M")
            total_params += real_tucker_params
        if hasattr(self, 'grid_features'):
            grid_params = sum(p.numel() for p in self.grid_features.parameters())
            print(f"  Feature Grid:    {grid_params / 1e6:.3f}M")
            total_params += grid_params
        if hasattr(self, 'complex_tucker'):
            complex_tucker_params = sum(p.numel() for p in self.complex_tucker.parameters())
            print(f"  Complex Tucker:  {complex_tucker_params / 1e6:.3f}M")
            total_params += complex_tucker_params
        if hasattr(self, 'forward_operators'):
            forward_params = sum(p.numel() for p in self.forward_operators.parameters())
            print(f"  Forward Operators:{forward_params / 1e6:.3f}M")
            total_params += forward_params
        if hasattr(self, 'backward_operators'):
            backward_params = sum(p.numel() for p in self.backward_operators.parameters())
            print(f"  Backward Operators:{backward_params / 1e6:.3f}M")
            total_params += backward_params
        if hasattr(self, 'upres'):
            upres_params = sum(p.numel() for p in self.upres.parameters())
            print(f"  Upsampling CNN:  {upres_params / 1e6:.3f}M")
            total_params += upres_params
        print(f"  Total:           {total_params / 1e6:.3f}M")

    def forward(self, norm_t, noise_op=None, zero_real_tucker=False, zero_feature_grid=False, zero_complex_tucker=False, zero_complex_grid=False, return_operators=False):
        if type(norm_t) is not torch.Tensor:
            norm_t = torch.tensor([norm_t], device=self.grid_features.grid.device, dtype=torch.float32)

        norm_t = (norm_t * (self.orig_T - 1) + float(self.operator_steps)) * self.dT
        zero_base = self._zero_base.expand(norm_t.shape[0], -1, -1, -1)

        B = norm_t.shape[0]
        S = int(self.operator_steps)

        offsets = self._operator_offsets.to(dtype=norm_t.dtype)
        times = (norm_t.unsqueeze(1) + offsets).clamp(0.0, 1.0)  # [B, 2S+1]
        times_flat = times.reshape(-1)

        # Batch-evaluate active modules once over all shifted times
        comps = {}

        if not zero_real_tucker and hasattr(self, 'real_tucker'):
            real_all = self.real_tucker(times_flat)  # [B*(2S+1), C, H, W]
            comps['real'] = real_all.view(B, 2 * S + 1, self.C, self.H, self.W)
        else:
            comps['real'] = zero_base.unsqueeze(1).expand(B, 2 * S + 1, -1, -1, -1)

        if not zero_feature_grid and hasattr(self, 'grid_features'):
            grid_all = self.grid_features(times_flat)
            comps['grid'] = grid_all.view(B, 2 * S + 1, self.C, self.H, self.W)
        else:
            comps['grid'] = zero_base.unsqueeze(1).expand(B, 2 * S + 1, -1, -1, -1)

        if hasattr(self, 'complex_tucker') and not (zero_complex_tucker and zero_complex_grid):
            complex_all = self.complex_tucker(
                times_flat,
                zero_tucker=zero_complex_tucker,
                zero_grid=zero_complex_grid,
            )
            comps['complex'] = complex_all.view(B, 2 * S + 1, self.C, self.H, self.W)
        else:
            comps['complex'] = zero_base.unsqueeze(1).expand(B, 2 * S + 1, -1, -1, -1)

        # Build concatenation in the same order heads were created so GroupNorm groups
        parts = [comps[name] for name in getattr(self, '_head_order', ['grid', 'real', 'complex'])]
        total_c = sum(p.shape[2] for p in parts)
        base_all = torch.cat(parts, dim=2)  # [B, 2S+1, C_total, H, W]
        base_all_flat = base_all.view(B * (2 * S + 1), total_c, self.H, self.W)
        base_all_gnorm = self.groupnorm(base_all_flat)
        base_all_gnorm = base_all_gnorm.view(B, 2 * S + 1, total_c, self.H, self.W)

        # center slice is the current input
        center_idx = S
        current_input = base_all_gnorm[:, center_idx]

        operator_residual = torch.zeros_like(current_input)

        for i in range(S):
            prev_idx = center_idx - (i + 1)
            next_idx = center_idx + (i + 1)

            prev_base = base_all_gnorm[:, prev_idx]
            next_base = base_all_gnorm[:, next_idx]

            forward_operator = self.forward_operators[i]
            forward_input = torch.cat([prev_base, current_input], dim=1)
            forward_prediction = forward_operator(forward_input, times[:, prev_idx])
            operator_residual += forward_prediction

            backward_operator = self.backward_operators[i]
            backward_input = torch.cat([current_input, next_base], dim=1)
            backward_prediction = backward_operator(backward_input, times[:, next_idx])
            operator_residual += backward_prediction

        aggregated = current_input + operator_residual
        refined = self.upres(aggregated)

        if return_operators:
            refined_forward = self.upres(forward_prediction)
            refined_backward = self.upres(backward_prediction)
            return refined, refined_forward, refined_backward
        return refined


def split_segments(total_frames, num_segments):
    if num_segments <= 1:
        return [(0, total_frames)]

    segment_size = math.ceil(total_frames / num_segments)
    ranges = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + segment_size)
        ranges.append((start, end))
        start = end
    return ranges


class MosaicNika(nn.Module):
    def __init__(self, target_shape, k, model_kwargs, out_channels, device='cuda', num_segments=1, compile_ops=False):
        super().__init__()
        self.num_segments = max(1, num_segments)
        self.total_frames = target_shape[3]
        self.segment_ranges = split_segments(self.total_frames, self.num_segments)

        self.models = nn.ModuleList()
        for start, end in self.segment_ranges:
            seg_length = max(1, end - start)
            seg_shape = [target_shape[0], target_shape[1], target_shape[2], seg_length]
            model = NikaBlock(
                target_shape=seg_shape,
                k=k,
                **model_kwargs,
                out_channels=out_channels,
                device=device,
            )
            self.models.append(model)

        self._device = torch.device(device)
        self._segment_size = math.ceil(self.total_frames / self.num_segments)
        self._total_frames_tensor = torch.tensor(self.total_frames, device=self._device, dtype=torch.long)

        total_params = sum(sum(p.numel() for p in model.parameters()) for model in self.models)
        # Count all parameters registered under this module (includes submodules)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MosaicNika: {len(self.models)} segments, total params: {total_params / 1e6:.3f}M")

    def make_segment_map(self, norm_t):
        if not isinstance(norm_t, torch.Tensor):
            norm_t = torch.tensor([norm_t], device=self._device, dtype=torch.float32)
        elif norm_t.device != self._device:
            norm_t = norm_t.to(self._device)
        norm_t = norm_t.view(-1)

        positions = (norm_t * max(self.total_frames - 1, 1)).clamp(0.0, self.total_frames - 1)
        frame_ids = (positions + 1e-4).floor().long()
        segment_ids = torch.div(frame_ids, self._segment_size, rounding_mode='floor').clamp(max=self.num_segments - 1)

        seg_start = segment_ids.to(dtype=positions.dtype) * float(self._segment_size)
        seg_end = torch.minimum((segment_ids + 1) * self._segment_size, self._total_frames_tensor).to(dtype=positions.dtype)
        seg_denom = (seg_end - seg_start - 1.0).clamp(min=1.0)
        local_norm = ((positions - seg_start) / seg_denom).clamp(0.0, 1.0)
        return segment_ids, local_norm

    def forward_segment(self, segment_id, local_norm_t, **kwargs):
        if self.num_segments == 1:
            return self.models[0](local_norm_t, **kwargs)

        if isinstance(segment_id, torch.Tensor):
            segment_id = int(segment_id.item())
        return self.models[segment_id](local_norm_t, **kwargs)

    def forward(self, norm_t, **kwargs):
        if self.num_segments == 1:
            return self.models[0](norm_t, **kwargs)

        segment_ids, local_norm = self.make_segment_map(norm_t)
        if segment_ids.numel() == 0:
            return torch.empty((0, 3, self.models[0].grid_features.grid.shape[1], self.models[0].grid_features.grid.shape[2]), device=self._device)

        outputs = None
        for segment_id in range(self.num_segments):
            mask = segment_ids == segment_id
            if not mask.any():
                continue
            segment_out = self.models[segment_id](local_norm[mask], **kwargs)
            if segment_out.ndim == 3:
                segment_out = segment_out.unsqueeze(0)

            if outputs is None:
                outputs = torch.empty(
                    (segment_ids.shape[0],) + segment_out.shape[1:],
                    device=segment_out.device,
                    dtype=segment_out.dtype,
                )

            outputs[mask] = segment_out

        return outputs


def feature_test(vid, dataset_name, model_name, config, device, batch_size=32, num_segments=None, operator_steps=None):
    base_name = os.path.basename(dataset_name)
    model_kwargs = REFERENCES[config]
    # copy and allow dataset-specific overrides
    model_kwargs = dict(model_kwargs)
    # double feature-grid channels for high-res 'bunny' dataset
    if base_name == 'bunny':
        model_kwargs['grid_ranks'] = model_kwargs['grid_ranks'] * 2

    # extract config default for segments and use CLI override if provided
    config_segments = model_kwargs.pop('num_segments', 1)
    if num_segments is None:
        num_segments = int(config_segments)
    else:
        num_segments = max(1, int(num_segments))
    print(f"Using num_segments={num_segments} (config default={config_segments})")

    # allow CLI override for operator_steps; otherwise keep config value
    if operator_steps is not None:
        model_kwargs['operator_steps'] = int(operator_steps)

    model = MosaicNika(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        model_kwargs=model_kwargs,
        out_channels=3,
        device=device,
        num_segments=num_segments,
    )
    segment_ranges = model.segment_ranges

    base_lr = 1e-2
    basis_params = []
    for sm in model.models:
        if hasattr(sm, 'real_tucker'):
            basis_params.extend(list(sm.real_tucker.parameters()))
        if hasattr(sm, 'complex_tucker'):
            basis_params.extend(list(sm.complex_tucker.parameters()))
        if hasattr(sm, 'grid_features'):
            basis_params.extend(list(sm.grid_features.parameters()))
    basis_ids = set(map(id, basis_params))
    rest_params = [p for p in model.parameters() if id(p) not in basis_ids]
    opt = SOAP([
        {"params": basis_params, "lr": base_lr},
        {"params": rest_params, "lr": base_lr},
    ], lr=base_lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='max',
        factor=0.5,
        patience=40,
        threshold = 0.015,
        threshold_mode='abs',
        cooldown=20,
        min_lr=2e-3,
    )

    best_psnr = float('-inf')
    best_epoch = -1
    current_batch_size = batch_size
    epoch = 0

    while epoch < 2000:
        try:
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            start_time = time.time()
            for segment_id, (seg_start, seg_end) in enumerate(segment_ranges):
                segment_length = seg_end - seg_start
                if segment_length == 0:
                    continue

                num_batches = (segment_length + current_batch_size - 1) // current_batch_size
                for b in range(num_batches):
                    min_t = seg_start + b * current_batch_size
                    max_t = min(seg_start + (b + 1) * current_batch_size, seg_end)
                    batch_gt = vid[min_t:max_t].to(torch.float32) / 255.0
                    t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.float32)
                    if segment_length > 1:
                        local_t = (t_batch - seg_start) / (segment_length - 1)
                    else:
                        local_t = torch.zeros_like(t_batch)

                    prediction = model.forward_segment(segment_id, local_t)

                    mse = F.mse_loss(prediction, batch_gt)
                    psnr = -10.0 * torch.log10(mse + 1e-8)

                    frame_loss = (-psnr) * (batch_gt.shape[0] / vid.shape[0])  # weight by number of frames in batch
                    frame_loss.backward()
                    loss += frame_loss
            average_frame_time = (time.time() - start_time) / vid.shape[0]
            epoch_psnr = -loss.item()
            print(f"[{dataset_name}] Epoch {epoch} loss: {loss.item():.4f}, time: {average_frame_time:.5f}s, PSNR: {epoch_psnr:.2f}")

            if epoch_psnr > best_psnr and (epoch - best_epoch >= 10 or best_epoch == -1):
                best_psnr = epoch_psnr
                best_epoch = epoch
                os.makedirs("models", exist_ok=True)
                model_path = f"models/{config}-{model_name}-epoch{epoch}-psnr{best_psnr:.2f}.torch"
                torch.save(model.state_dict(), model_path)
                os.sync()
                print(f"[{dataset_name}] New best model saved at epoch {epoch} with PSNR: {best_psnr:.2f}")

            opt.step()
            scheduler.step(epoch_psnr)

            if epoch % 25 == 0:
                print(f"[{dataset_name}] Epoch {epoch}: Tucker PSNR: {epoch_psnr:.2f}")

            epoch += 1
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                if current_batch_size <= 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                print(f"OOM during epoch {epoch}; reducing batch_size to {current_batch_size} and retrying.")
                continue
            raise

    print(f"Best PSNR achieved: {best_psnr:.2f} at epoch {best_epoch}")


def run_all_feature_tests(names, basedir, config, device, batch_size=32, num_segments=None, operator_steps=None):
    for dataset_name in names:
        print(f"\n=== Starting dataset: {dataset_name} ===")
        vid = load_video_frames(f"{basedir}/{dataset_name}", device, dtype=torch.uint8, normalize=False)
        torch.manual_seed(45)
        feature_test(
            vid,
            dataset_name=dataset_name,
            model_name=os.path.basename(dataset_name),
            config=config,
            device=device,
            batch_size=batch_size,
            num_segments=num_segments,
            operator_steps=operator_steps,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nika feature test")
    parser.add_argument("--basedir", default="static/benchmarks", help="Base video directory")
    parser.add_argument("--name", default="bunny", help="Single video name")
    parser.add_argument("--config", default="small", help="Config name from configs.REFERENCES")
    parser.add_argument("--device", default="cuda:0", help="Device to run on, e.g. cuda:0 or 0")
    parser.add_argument("--batch_size", type=int, default=32, help="Initial training batch size")
    parser.add_argument("--segments", type=int, default=None, help="Number of temporal model shards for mosaic mode (overrides config)")
    parser.add_argument("--operator_steps", type=int, default=None, help="Number of operator steps (overrides config)")
    args = parser.parse_args()

    device = args.device
    if isinstance(device, str):
        if re.fullmatch(r"\d+", device):
            device = f"cuda:{device}"
        elif re.fullmatch(r"cuda\d+", device):
            device = device.replace('cuda', 'cuda:')

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    all_names = [
        "uvg/honey",
        "uvg/bosphorus",
        "uvg/beauty",
        # "uvg/jockey",
        "uvg/ready",
        # "uvg/shake",
        "uvg/yacht",
    ]

    if args.name == "all":
        names = all_names
    else:
        names = [args.name]

    run_all_feature_tests(
        names,
        args.basedir,
        args.config,
        device,
        batch_size=args.batch_size,
        num_segments=args.segments,
        operator_steps=args.operator_steps,
    )
