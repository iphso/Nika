import os
import time
import math
import glob
from functools import partial

import torch
from torch.profiler import profile, record_function, ProfilerActivity
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
        self.max_chunk_size = 500
        self.target_dim = target_dim
        self.rank = rank
        self.is_complex = is_complex
        self.device = device

        def make_chunk(chunk_size):
            if self.is_complex:
                return nn.Parameter(torch.randn(chunk_size, rank, device=device) * base_mag), \
                          nn.Parameter(torch.zeros(chunk_size, rank, device=device))  # real, imag
            else:
                return nn.Parameter(torch.randn(chunk_size, rank, device=device) * base_mag)
        num_chunks = int((target_dim - 1) // self.max_chunk_size) + 1
        self.chunked = False
        if num_chunks > 1:
            self.chunked = True

        if self.chunked:
            if self.is_complex:
                self.real_chunks = nn.ParameterList()
                self.imag_chunks = nn.ParameterList()
            else:
                self.chunks = nn.ParameterList()

            for i in range(num_chunks):
                start = i * self.max_chunk_size
                end = min((i + 1) * self.max_chunk_size, target_dim)
                chunk_size = end - start
                if self.is_complex:
                    real_param, imag_param = make_chunk(chunk_size)
                    self.real_chunks.append(real_param)
                    self.imag_chunks.append(imag_param)
                else:
                    param = make_chunk(chunk_size)
                    self.chunks.append(param)
        else:
            if self.is_complex:
                self.U_real = nn.Parameter(torch.randn(target_dim, rank, device=device) * base_mag)
                self.U_imag = nn.Parameter(torch.zeros(target_dim, rank, device=device))
            else:
                self.U = nn.Parameter(torch.randn(target_dim, rank, device=device) * base_mag)

    def forward(self):
        if self.chunked:
            if self.is_complex:
                real_parts = []
                imag_parts = []
                for r_chunk, i_chunk in zip(self.real_chunks, self.imag_chunks):
                    real_parts.append(r_chunk)
                    imag_parts.append(i_chunk)
                U_real = torch.cat(real_parts, dim=0)
                U_imag = torch.cat(imag_parts, dim=0)
                U = torch.complex(U_real, U_imag)
            else:
                parts = []
                for chunk in self.chunks:
                    parts.append(chunk)
                U = torch.cat(parts, dim=0)
        else:
            if self.is_complex:
                U = torch.complex(self.U_real, self.U_imag)
            else:
                U = self.U
        return U

    def get(self, target):
        U = self.forward()
        target = torch.as_tensor(target, device=U.device, dtype=torch.float32)

        t_norm = 2.0 * target - 1.0  # [-1, 1]
        t_norm = t_norm.view(1, -1, 1, 1)

        grid = torch.zeros((1, t_norm.shape[1], 1, 2), device=U.device, dtype=t_norm.dtype)
        grid[..., 1] = t_norm.squeeze(-1)  # y coord (H)
        # x coord (W=1) stays 0

        def _sample(inp):
            inp_ = inp.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # [1, R, T, 1]
            out = F.grid_sample(inp_, grid, mode="bilinear", align_corners=True, padding_mode="border")
            return out.squeeze(0).squeeze(-1).transpose(0, 1)  # [B, R]

        if torch.is_complex(U):
            return torch.complex(_sample(U.real), _sample(U.imag))
        return _sample(U)


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

    def __init__(self, target_shape, ranks, device='cuda'):
        super().__init__(target_shape, ranks, device=device)
        half_W = (self.W // 2) + 1
        self.UH = TuckerFactor(self.H, self.rH, is_complex=True, device=device)
        self.UW = TuckerFactor(half_W, self.rW, is_complex=True, device=device)
        self.UC = TuckerFactor(self.C, self.rC, is_complex=True, device=device)
        self.UT = TuckerFactor(self.T, self.rT, is_complex=True, device=device)

        self.G = None  # override parent
        self.G_real = nn.Parameter(torch.randn(self.rT, self.rC, self.rH, self.rW, device=device) * 1e-2)
        self.G_imag = nn.Parameter(torch.zeros(self.rT, self.rC, self.rH, self.rW, device=device))

        self.feature_grid = FeatureGrid([self.C * 2, self.H, half_W, self.T], grid_res=[self.C * 2, self.H, half_W, 1], device=device)

    def forward(self, t):
        UH = self.UH()
        UW = self.UW()
        UC = self.UC()
        UT = self.UT.get(t)
        G = torch.complex(self.G_real, self.G_imag)
        construct = tucker_construct(UT, UC, UH, UW, G)

        grid = self.feature_grid(t)
        complex_grid = torch.complex(*grid.chunk(2, dim=1))
        construct = construct * complex_grid
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
        self.grid_c = grid_res[0]
        self.grid_h = grid_res[1]
        self.grid_w = grid_res[2]
        self.grid_t = grid_res[3]

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
        device = self.grid.device
        B = t.shape[0]

        sample_grid3 = torch.empty((B, self.H, self.W, 3), device=device, dtype=self._xy_base.dtype)
        sample_grid3[..., :2] = self._xy_base
        sample_grid3[..., 2] = (2.0 * t - 1.0).view(B, 1, 1)

        sample_grid3 = sample_grid3.unsqueeze(1)  # [B,1,H,W,3]

        grid_5d = self._5d_grid().expand(B, -1, -1, -1, -1)

        sampled = F.grid_sample(
            grid_5d,               # [B, C, T_g, H_g, W_g]
            sample_grid3,           # [B, 1, H_out, W_out, 3]
            mode='bilinear',
            align_corners=False,
            padding_mode='border',
        )  # → [B, C, 1, H_out, W_out]

        result = sampled.squeeze(2)
        if hasattr(self, 'channel_proj'):
            result = self.channel_proj(result.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return result.contiguous()


def tucker_construct(UT, UC, UH, UW, G):
    UT = UT.contiguous()
    UC = UC.contiguous()
    UH = UH.contiguous()
    UW = UW.contiguous()
    G = G.contiguous()

    def _col_norm(M, eps=1e-8):
        if torch.is_complex(M):
            norms_sq = (M.real**2 + M.imag**2).sum(dim=0, keepdim=True)
            norms = torch.sqrt(norms_sq + eps)
        else:
            norms = M.norm(dim=0, keepdim=True) + eps
        return M / norms

    UH = _col_norm(UH)
    UW = _col_norm(UW)
    UC = _col_norm(UC)
    UT = _col_norm(UT)

    X = torch.einsum('ijkl,ti,cj,hk,wl->tchw', G, UT, UC, UH, UW)
    return X


class BasicUpres(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, k, encoding_len=64, device='cuda'):
        super().__init__()
        half_k = k // 2
        self.k = k

        self.upres = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=1),
            nn.PixelShuffle(upscale_factor=k),
        ).to(device)

        #kaiming init
        for m in self.upres.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        base = self.upres(x)
        return base


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
        if x.all() == 0:
            return torch.zeros(
                (x.shape[0], self.out_channels, x.shape[2], x.shape[3]),
                device=self.device,
                dtype=x.dtype,
            )
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
    def __init__(self, target_shape, k, real_tucker_ranks, complex_tucker_ranks, grid_ranks, conv_hidden, out_channels, device):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.H = int(self.H // k); self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]
        self.dT = 1.0 / (self.T - 1)
        self.real_tucker = RealTucker(
            target_shape=self.internal_shape,
            ranks=real_tucker_ranks,
            device=device,
        )
        self.real_tucker = torch.compile(self.real_tucker)

        self.grid_features = FeatureGrid(
            target_shape=self.internal_shape,
            grid_res=grid_ranks,
            device=device,
        )
        self.grid_features = torch.compile(self.grid_features)

        self.complex_tucker = ComplexTucker(
            target_shape=self.internal_shape,
            ranks=complex_tucker_ranks,
            device=device,
        )

        self.n_heads = 3

        self.groupnorm = nn.GroupNorm(num_groups=self.n_heads, num_channels=self.n_heads * self.C).to(device)
        self.groupnorm = torch.compile(self.groupnorm)

        op_hdim = 64
        self.operator_steps = 2

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
            self.forward_operators.append(torch.compile(fwd))
            self.backward_operators.append(torch.compile(bwd))

        self.upres = BasicUpres(
            in_channels = self.n_heads * self.C,
            out_channels = out_channels,
            hidden = conv_hidden,
            k = k,    
            device = device,
        )
        self.upres = torch.compile(self.upres)

        self.log_stats()

    def log_stats(self):
        real_tucker_params = sum(p.numel() for p in self.real_tucker.parameters())
        complex_tucker_params = sum(p.numel() for p in self.complex_tucker.parameters())
        grid_params = sum(p.numel() for p in self.grid_features.parameters())
        upres_params = sum(p.numel() for p in self.upres.parameters())
        operator_params = sum(p.numel() for p in self.forward_operators.parameters()) + sum(p.numel() for p in self.backward_operators.parameters())
        total_params = real_tucker_params + complex_tucker_params + grid_params + upres_params + operator_params
        print(f"NikaBlock parameters:")
        print(f"  Real Tucker:     {real_tucker_params / 1e6:.3f}M")
        print(f"  Complex Tucker:  {complex_tucker_params / 1e6:.3f}M")
        print(f"  Feature Grid:    {grid_params / 1e6:.3f}M")
        print(f"  Forward Operator:{sum(p.numel() for p in self.forward_operators.parameters()) / 1e6:.3f}M")
        print(f"  Backward Operator:{sum(p.numel() for p in self.backward_operators.parameters()) / 1e6:.3f}M")
        print(f"  Upsampling CNN:  {upres_params / 1e6:.3f}M")
        print(f"  Total:           {total_params / 1e6:.3f}M")

    def forward(self, norm_t, noise_op=None, zero_real_tucker=False, zero_complex_tucker=False, zero_feature_grid=False, return_operators=False):
        if type(norm_t) is not torch.Tensor:
            norm_t = torch.tensor([norm_t], device=self.grid_features.grid.device, dtype=torch.float32)

        if not hasattr(self, "_zero_base"):
            self.register_buffer(
            "_zero_base",
            torch.zeros(1, self.C, self.H, self.W, device=norm_t.device, dtype=torch.float32),
            persistent=False,
        )

        zero_base = self._zero_base.expand(norm_t.shape[0], -1, -1, -1)
        curr_real_tucker = self.real_tucker(norm_t) if not zero_real_tucker else zero_base
        curr_real_grid = self.grid_features(norm_t) if not zero_feature_grid else zero_base
        curr_complex_tucker = self.complex_tucker(norm_t) if not zero_complex_tucker else zero_base

        current_base = torch.cat([curr_real_grid, curr_real_tucker, curr_complex_tucker], dim=1)
        current_input = self.groupnorm(current_base)

        operator_residual = torch.zeros_like(current_input)
        for i in range(self.operator_steps):
            step_len = (i + 1) * self.dT
            mask_prev = (norm_t >= step_len)
            norm_t_prev = (norm_t[mask_prev] - step_len) if mask_prev.any() else None
            mask_next = (norm_t <= (1 - step_len))
            norm_t_next = (norm_t[mask_next] + step_len) if mask_next.any() else None

            prev_real_tucker, next_real_tucker = zero_base, zero_base
            prev_complex_tucker, next_complex_tucker = zero_base, zero_base
            prev_grid, next_grid = zero_base, zero_base

            if not zero_real_tucker:
                prev_real_tucker = self.real_tucker(norm_t_prev) if mask_prev.any() else zero_base
                next_real_tucker = self.real_tucker(norm_t_next) if mask_next.any() else zero_base
            
            if not zero_feature_grid:
                prev_grid = self.grid_features(norm_t_prev) if mask_prev.any() else zero_base
                next_grid = self.grid_features(norm_t_next) if mask_next.any() else zero_base

            if not zero_complex_tucker:
                prev_complex_tucker = self.complex_tucker(norm_t_prev) if mask_prev.any() else zero_base
                next_complex_tucker = self.complex_tucker(norm_t_next) if mask_next.any() else zero_base

            prev_base = torch.cat([prev_grid, prev_real_tucker, prev_complex_tucker], dim=1)
            prev_base = self.groupnorm(prev_base)

            prev_frames = torch.zeros_like(current_input)
            prev_frames[mask_prev] = prev_base
            forward_operator = self.forward_operators[i]
            if mask_prev.any():
                forward_prediction = forward_operator(torch.cat([prev_frames, current_input], dim=1), norm_t_prev)
                operator_residual += forward_prediction

            next_base = torch.cat([next_grid, next_real_tucker, next_complex_tucker], dim=1)
            next_base = self.groupnorm(next_base)
            next_frames = torch.zeros_like(current_input)
            next_frames[mask_next] = next_base
            backward_operator = self.backward_operators[i]
            if mask_next.any():
                backward_prediction = backward_operator(torch.cat([current_input, next_frames], dim=1), norm_t_next)
                operator_residual += backward_prediction

        aggregated = current_input + operator_residual
        refined = self.upres(aggregated)

        if return_operators:
            refined_forward = self.upres(forward_prediction)
            refined_backward = self.upres(backward_prediction)
            return refined, refined_forward, refined_backward
        return refined

    def test_images(self, output_dir):
        # self.eval()
        with torch.no_grad():
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            rand_vals = torch.linspace(0, self.T - 1, steps=10, dtype=torch.int64, device=self.grid_features.grid.device)
            torch.cuda.synchronize()
            start_time = time.time()
            imgs = self.forward(rand_vals, rand_vals)
            torch.cuda.synchronize()
            average_frame_time = (time.time() - start_time) / rand_vals.shape[0]
            print(f"Average inference time per frame: {average_frame_time:.5f}s")
            print(f"FPS: {1.0 / average_frame_time:.2f}")
            for i in range(imgs.shape[0]):
                img = imgs[i].clamp(0.0, 1.0)
                save_image(img, f"{output_dir}/frame_{i:04d}.png")


def feature_test(vid, name, config, device):
    batch_size = 6
    model_kwargs = REFERENCES[config]
    model = NikaBlock(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        **model_kwargs,
        out_channels=3,
        device=device,
    )

    # create optimizer with two parameter groups:
    #  - basis_params: the two tuckers + the feature grid (we'll decay their lr)
    #  - rest_params: all remaining parameters (keep their lr constant)
    base_lr = 1e-2
    basis_params = list(model.real_tucker.parameters()) + list(model.complex_tucker.parameters()) + list(model.grid_features.parameters())
    basis_ids = set(map(id, basis_params))
    rest_params = [p for p in model.parameters() if id(p) not in basis_ids]
    opt = SOAP([
        {"params": basis_params, "lr": base_lr},
        {"params": rest_params, "lr": base_lr},
    ], lr=base_lr)

    best_psnr = float('-inf')
    best_epoch = -1

    for epoch in range(2000):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        start_time = time.time()
        num_batches = (vid.shape[0] + batch_size - 1) // batch_size
        for t in range(num_batches):
            min_t = t * batch_size
            max_t = min((t + 1) * batch_size, vid.shape[0])
            batch_gt = vid[min_t:max_t].to(torch.float32) / 255.0
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            norm_t_batch = t_batch.float() / (vid.shape[0] - 1)
            prediction = model(norm_t_batch)
            mse = F.mse_loss(prediction, batch_gt)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            frame_loss = (-psnr).mean() / num_batches
            frame_loss.backward()
            loss += frame_loss
        average_frame_time = (time.time() - start_time) / vid.shape[0]
        epoch_psnr = -loss.item()
        print(f"Epoch {epoch} loss: {loss.item():.4f}, time: {average_frame_time:.5f}s, PSNR: {epoch_psnr:.2f}")

        # schedule: linearly anneal basis lr from base_lr -> 0 over epochs [500, 1500]
        if epoch < 500:
            new_basis_lr = base_lr
        elif epoch <= 1500:
            frac = (epoch - 500) / float(1500 - 500)
            new_basis_lr = base_lr * (1.0 - frac)
        else:
            new_basis_lr = 0.0
        # our first param_group is the basis group
        try:
            opt.param_groups[0]["lr"] = new_basis_lr
        except Exception:
            pass

        if epoch_psnr > best_psnr and (epoch - best_epoch >= 10 or best_epoch == -1):
            best_psnr = epoch_psnr
            best_epoch = epoch
            model_path = f"models/{config}-{name}-epoch{epoch}-psnr{best_psnr:.2f}.torch"
            torch.save(model.state_dict(), model_path)
            os.sync()
            print(f"New best model saved at epoch {epoch} with PSNR: {best_psnr:.2f}")

        opt.step()

        if epoch % 25 == 0:
            print(f"Epoch {epoch}: Tucker PSNR: {epoch_psnr:.2f}")
            model.test_images("out_feature_test")

    print(f"Best PSNR achieved: {best_psnr:.2f} at epoch {best_epoch}")


if __name__ == "__main__":
    device = "cuda:0"
    name = "shake"
    torch.manual_seed(42)
    vid = load_video_frames(f"static/benchmarks/uvg/{name}", device, max_frames=600, dtype=torch.uint8, normalize=False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    feature_test(vid, name, f"large", device=device)