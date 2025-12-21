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


def load_image(png_path, normalize=True, device='cuda', dtype=torch.float32):
    png = iio.imread(png_path, plugin="pillow")  # H,W,3 uint8
    if normalize:
        png = png.astype(np.float32) / 255.0
    x = png.permute(2,0,1).contiguous()            # 3,H,W (CHW)
    x = x.unsqueeze(0)                           # 1,3,H,W (BCHW)
    return x.to(dtype)


def hermitian_project_shifted(Ks):
    B, C, kh, kw = Ks.shape
    Ku = torch.fft.ifftshift(Ks, dim=(-2, -1))
    # Partner mapping in UNshifted coords is i → (-i mod k), which equals flip + roll(+1)
    partner = torch.roll(
        torch.roll(torch.flip(Ku, dims=(-2, -1)), shifts=1, dims=-2),
        shifts=1, dims=-1
    ).conj()

    Ksym = 0.5 * (Ku + partner)

    # Force self-conjugate bins to be real
    Ksym[..., 0, 0] = Ksym[..., 0, 0].real + 0j
    if kw % 2 == 0:
        mid = kw // 2
        Ksym[..., 0,  mid] = Ksym[..., 0,  mid].real + 0j
        Ksym[..., mid, 0 ] = Ksym[..., mid, 0 ].real + 0j
        Ksym[..., mid, mid] = Ksym[..., mid, mid].real + 0j
    return torch.fft.fftshift(Ksym, dim=(-2, -1))  # back to SHIFTED


def load_video_frames(
    dir_path,
    device="cuda",
    dtype=torch.float32,
    max_frames=100,
    normalize=True,

):
    torch.cuda.set_device(device)
    all_paths = sorted(glob.glob(f"{dir_path}/*.png"))
    if not all_paths:
        raise RuntimeError(f"No frames found in {dir_path}")
    if len(all_paths) <= max_frames:
        paths = all_paths
    else:
        # randomly select max_frames without replacement
        paths = all_paths[:max_frames]
        # paths = random.sample(all_paths, max_frames)
    # Read one to get H, W
    first = iio.imread(paths[0], plugin="pillow")
    H, W = first.shape[:2]

    # Preallocate CPU tensor (pinned for faster HtoD)
    vid_cpu = torch.empty((len(paths), 3, H, W), dtype=dtype, pin_memory=True)

    # Fill preallocated buffer
    for t, p in enumerate(paths):
        if t % 50 == 0:
            print(f"Loading frame {t}/{len(paths)}")
        if normalize:
            arr = iio.imread(p, plugin="pillow").astype("float32") / 255.0
        else:
            arr = iio.imread(p, plugin="pillow")
        arr = torch.from_numpy(arr)
        lin = arr.permute(2, 0, 1).contiguous()         # 3,H,W
        # save_image(lin, f"frame_{t:04d}.png")
        vid_cpu[t].copy_(lin)

    # Single transfer to GPU; cast dtype here
    return vid_cpu.to(device=device, dtype=dtype, non_blocking=True)


def _to_energy_map(G):  # G: [rH,rW,rC] or [rH,rW,rC,rT], real or complex
    if G.is_complex():
        E = G.real.pow(2) + G.imag.pow(2)
    else:
        E = G.pow(2)
    # reduce over channel (and time if present)
    if E.ndim == 4:
        E = E.mean(dim=(2, 3))  # [rH, rW]
    else:
        E = E.mean(dim=2)       # [rH, rW]
    return E  # energy map over (rH, rW)


def block_scores_pool_from_core(G, bh: int, bw: int):
    # Build block scores from core G without time-mix
    # Scores = avg_pool over energy map
    E = _to_energy_map(G)  # [rH, rW]
    rH, rW = E.shape
    nBH = math.ceil(rH / bh)
    nBW = math.ceil(rW / bw)
    padH = nBH * bh - rH
    padW = nBW * bw - rW
    if padH or padW:
        E = F.pad(E, (0, padW, 0, padH), value=0)
    S = F.avg_pool2d(E.unsqueeze(0).unsqueeze(0), kernel_size=(bh, bw), stride=(bh, bw))
    return S.squeeze(0).squeeze(0)  # [nBH, nBW]


def topk_block_mask_from_scores(scores: torch.Tensor, keep_ratio: float):
    K = max(1, int(round(scores.numel() * keep_ratio)))
    vals, _ = torch.topk(scores.flatten(), K, largest=True)
    thresh = vals.min()
    return (scores >= thresh)


class SparseBlockCore(nn.Module):
    """
    Fixed block-sparse mask for G (applies to both real/imag).
    Build once (static), or call .refresh(...) periodically.
    """
    def __init__(self, G_real, G_imag=None, block_size=(4,4), keep_ratio=0.10):
        super().__init__()
        bh, bw = block_size
        with torch.no_grad():
            G = torch.complex(G_real, G_imag) if G_imag is not None else G_real
            scores = block_scores_pool_from_core(G, bh, bw)               # [nBH, nBW]
            block_mask = topk_block_mask_from_scores(scores, keep_ratio)  # [nBH, nBW]

            # expand to [rH, rW]
            full = block_mask.repeat_interleave(bh, 0).repeat_interleave(bw, 1)
            full = full[:G_real.shape[0], :G_real.shape[1]]               # crop to exact size

        # register as buffer on same device/dtype as G_real
        self.register_buffer("mask_hw", full.to(device=G_real.device))
        self.block_size = block_size

    @torch.no_grad()
    def refresh(self, G_real, G_imag=None, keep_ratio=None):
        """Recompute mask from current core (optionally change keep_ratio)."""
        if keep_ratio is None:
            keep_ratio = (self.mask_hw.float().mean().item())
        bh, bw = self.block_size
        G = torch.complex(G_real, G_imag) if G_imag is not None else G_real
        scores = block_scores_pool_from_core(G, bh, bw)
        block_mask = topk_block_mask_from_scores(scores, keep_ratio)
        full = block_mask.repeat_interleave(bh, 0).repeat_interleave(bw, 1)
        full = full[:G_real.shape[0], :G_real.shape[1]]
        self.mask_hw.copy_(full.to(self.mask_hw.device))

    def apply(self, G_real, G_imag=None):
        # Broadcast [rH,rW] -> [rH,rW,1,1] for [rH,rW,rC,rT]
        m = self.mask_hw
        while m.ndim < G_real.ndim:
            m = m.unsqueeze(-1)
        if G_imag is None:
            return G_real * m
        else:
            return (G_real * m, G_imag * m)


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
        print(f"TuckerFactor: target_dim={target_dim}, rank={rank}, num_chunks={num_chunks}, is_complex={is_complex}")
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

    def numel(self):
        if self.chunked:
            if self.is_complex:
                total = 0
                for r_chunk, i_chunk in zip(self.real_chunks, self.imag_chunks):
                    total += r_chunk.numel() + i_chunk.numel()
                return total
            else:
                total = 0
                for chunk in self.chunks:
                    total += chunk.numel()
                return total
        else:
            if self.is_complex:
                return self.U_real.numel() + self.U_imag.numel()
            else:
                return self.U.numel()
    
    def grad_norm(self):
        # Check that grads are turned on everywhere:
        for param in self.parameters():
            if not param.requires_grad:
                raise ValueError("Gradients are not enabled for all parameters.")
        total_norm = 0.0
        if self.chunked:
            if self.is_complex:
                for r_chunk, i_chunk in zip(self.real_chunks, self.imag_chunks):
                    if r_chunk.grad is not None:
                        total_norm += r_chunk.grad.norm().item() ** 2
                    if i_chunk.grad is not None:
                        total_norm += i_chunk.grad.norm().item() ** 2
            else:
                for chunk in self.chunks:
                    if chunk.grad is not None:
                        total_norm += chunk.grad.norm().item() ** 2
        else:
            if self.is_complex:
                if self.U_real.grad is not None:
                    total_norm += self.U_real.grad.norm().item() ** 2
                if self.U_imag.grad is not None:
                    total_norm += self.U_imag.grad.norm().item() ** 2
            else:
                if self.U.grad is not None:
                    total_norm += self.U.grad.norm().item() ** 2
        return math.sqrt(total_norm)

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
    
    def get_section(self, target, x_range=1, pad_mode='zero'):
        # Assume target is 0/1 normed index; convert to actual index
        target = int(target * (self.target_dim))
        half_range = (x_range - 1) // 2
        min_idx = max(target - half_range, 0)
        min_idx = min(min_idx, (self.target_dim - 1))  # for specific edge cases
        max_idx = min(target + half_range, (self.target_dim - 1))

        if self.chunked:
            start_chunk = min_idx // self.max_chunk_size
            start_rel_idx = min_idx % self.max_chunk_size
            end_chunk = max_idx // self.max_chunk_size
            end_rel_idx = max_idx % self.max_chunk_size

            start_chunk = int(start_chunk)
            end_chunk = int(end_chunk)
            start_rel_idx = int(start_rel_idx)
            end_rel_idx = int(end_rel_idx)

            if start_chunk == end_chunk:
                if self.is_complex:
                    r_chunk = self.real_chunks[start_chunk][start_rel_idx:end_rel_idx + 1]
                    i_chunk = self.imag_chunks[start_chunk][start_rel_idx:end_rel_idx + 1]
                    U_section = torch.complex(r_chunk, i_chunk)
                else:
                    U_section = self.chunks[start_chunk][start_rel_idx:end_rel_idx + 1]
            else:
                # Collect all relevant slices across chunks
                if self.is_complex:
                    r_chunks = []
                    i_chunks = []
                    for chunk_idx in range(start_chunk, end_chunk + 1):
                        if chunk_idx == start_chunk:
                            r_chunks.append(self.real_chunks[chunk_idx][start_rel_idx:])
                            i_chunks.append(self.imag_chunks[chunk_idx][start_rel_idx:])
                        elif chunk_idx == end_chunk:
                            r_chunks.append(self.real_chunks[chunk_idx][:end_rel_idx + 1])
                            i_chunks.append(self.imag_chunks[chunk_idx][:end_rel_idx + 1])
                        else:
                            r_chunks.append(self.real_chunks[chunk_idx])
                            i_chunks.append(self.imag_chunks[chunk_idx])
                    U_section = torch.complex(torch.cat(r_chunks, dim=0), torch.cat(i_chunks, dim=0))
                else:
                    chunks = []
                    for chunk_idx in range(start_chunk, end_chunk + 1):
                        if chunk_idx == start_chunk:
                            chunks.append(self.chunks[chunk_idx][start_rel_idx:])
                        elif chunk_idx == end_chunk:
                            chunks.append(self.chunks[chunk_idx][:end_rel_idx + 1])
                        else:
                            chunks.append(self.chunks[chunk_idx])
                    U_section = torch.cat(chunks, dim=0)
        else:
            U_section = self.forward()[min_idx:max_idx + 1]

        pad_len = x_range - (max_idx - min_idx + 1)
        if pad_len > 0:
            if target == 0:
                pad_left = pad_len
                pad_right = 0
            elif target >= self.target_dim - 1:
                pad_left = 0
                pad_right = pad_len

            if pad_mode == 'zero':
                U_section = F.pad(U_section, (0, 0, pad_left, pad_right), mode='constant', value=0)
            elif pad_mode == 'replicate':
                U_section = F.pad(U_section, (0, 0, pad_left, pad_right), mode='replicate')
        return U_section


class RealTucker(nn.Module):
    def __init__(self, target_shape, ranks, density=0.1, device='cuda'):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.rC, self.rH, self.rW, self.rT = ranks

        self.UH = TuckerFactor(self.H, self.rH, is_complex=False, device=device)
        self.UW = TuckerFactor(self.W, self.rW, is_complex=False, device=device)
        self.UC = TuckerFactor(self.C, self.rC, is_complex=False, device=device)
        self.UT = TuckerFactor(self.T, self.rT, is_complex=False, device=device)

        self.G = nn.Parameter(torch.randn(self.rH, self.rW, self.rC, self.rT, device=device) * 1e-2)

        if density < 1.0:
            self.sparse_core = SparseBlockCore(self.G, block_size=(4,4), keep_ratio=density)
            self.density = density

    def log_stats(self):
        print(f"Ranks: rC={self.rC}, rH={self.rH}, rW={self.rW}, rT={self.rT}")
        uh_params = self.UH.numel()
        uw_params = self.UW.numel()
        uc_params = self.UC.numel()
        ut_params = self.UT.numel()
        g_params = self.G.numel()

        if hasattr(self, 'density'):
            g_params = int(g_params * self.density)

        print(
            "Component params: "
            f"UH={uh_params}, "
            f"UW={uw_params}, "
            f"UC={uc_params}, "
            f"UT={ut_params}, "
            f"G={g_params}, "
            f"Total={uh_params + uw_params + uc_params + ut_params + g_params}"
        )

    def sparsify_core(self):
        if not hasattr(self, 'sparse_core'):
            self.sparse_core = SparseBlockCore(self.G, block_size=(4,4), keep_ratio=self.density)
        else:
            self.sparse_core.refresh(self.G, keep_ratio=self.density)

    def forward(self, t, t_range=3, target_H=None, target_W=None):
        UH = self.UH()
        UW = self.UW()
        UC = self.UC()
        UT = self.UT.get_section(t, x_range=t_range, pad_mode='zero')

        if hasattr(self, 'sparse_core'):
            self.sparse_core.refresh(self.G, keep_ratio=self.density)
            G = self.sparse_core.apply(self.G)
        else:
            G = self.G

        return tucker_construct(UT, UC, UH, UW, G, target_H=target_H, target_W=target_W)

    def full_image(self, t, t_range=3, H=None, W=None):
        output = self.forward(t, t_range=t_range, target_H=H, target_W=W)
        return output.contiguous()


class ComplexTucker(RealTucker):

    def __init__(self, target_shape, ranks, density=0.1, device='cuda'):
        super().__init__(target_shape, ranks, density=density, device=device)
        self.UH = TuckerFactor(self.H, self.rH, is_complex=True, device=device)
        self.UW = TuckerFactor(self.W, self.rW, is_complex=True, device=device)
        self.UC = TuckerFactor(self.C, self.rC, is_complex=True, device=device)
        self.UT = TuckerFactor(self.T, self.rT, is_complex=True, device=device)

        self.G_real = nn.Parameter(torch.randn(self.rH, self.rW, self.rC, self.rT, device=device) * 1e-2)
        self.G_imag = nn.Parameter(torch.zeros(self.rH, self.rW, self.rC, self.rT, device=device))
    
    def forward(self, t, t_range=3, target_H=None, target_W=None):
        UH = self.UH()
        UW = self.UW()
        UC = self.UC()
        UT = self.UT.get_section(t, x_range=t_range, pad_mode='zero')

        if hasattr(self, 'sparse_core'):
            self.sparse_core.refresh(self.G_real, self.G_imag, keep_ratio=self.density)
            G_real, G_imag = self.sparse_core.apply(self.G_real, self.G_imag)
        else:
            G_real = self.G_real
            G_imag = self.G_imag

        G = torch.complex(G_real, G_imag)

        return tucker_construct(UT, UC, UH, UW, G, target_H=target_H, target_W=target_W)

    def full_image(self, t, t_range=3, H=None, W=None):
        pred = self.forward(t, t_range=t_range, target_H=H, target_W=W)
        base = hermitian_project_shifted(pred)
        tucker_base = torch.fft.ifft2(base, norm='ortho').real
        return tucker_base.contiguous()


class FeatureGrid(nn.Module):
    def __init__(self, target_shape, grid_res, zero_init=False, device="cuda"):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.grid_c = grid_res[0]
        self.grid_h = grid_res[1]
        self.grid_w = grid_res[2]
        self.grid_t = grid_res[3]

        self.grid = nn.Parameter(torch.zeros(self.grid_c, self.grid_h, self.grid_w, self.grid_t, device=device))
        self.channel_proj = nn.Linear(self.grid_c, self.C, bias=True).to(device)
        print(f"FeatureGrid params: {self.grid.numel()}")

    def forward(self, t, target_H=None, target_W=None):
        device = self.grid.device
        H_out = target_H if target_H is not None else self.H
        W_out = target_W if target_W is not None else self.W

        # Compute normalized time coordinate
        t_norm = 2.0 * (float(t) / (self.T - 1)) - 1.0

        # Compute normalized spatial coordinates
        y_lin = torch.linspace(0, self.H - 1, steps=H_out, device=device)
        x_lin = torch.linspace(0, self.W - 1, steps=W_out, device=device)
        y_norm = 2.0 * (y_lin / (self.H - 1)) - 1.0
        x_norm = 2.0 * (x_lin / (self.W - 1)) - 1.0

        # Create sampling grid for a single time value
        y, x = torch.meshgrid(y_norm, x_norm, indexing='ij')
        t_grid = torch.full_like(y, t_norm)
        sample_grid = torch.stack((x, y, t_grid), dim=-1)  # [H_out, W_out, 3]
        sample_grid = sample_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H_out, W_out, 3]

        grid_5d = self.grid.permute(0, 3, 1, 2).unsqueeze(0)  # [1, C, T_g, H_g, W_g]

        sampled = F.grid_sample(
            grid_5d,               # [1, C, T_g, H_g, W_g]
            sample_grid,           # [1, 1, H_out, W_out, 3]
            mode='bilinear',
            align_corners=True,
            padding_mode='border',
        )  # → [1, C, 1, H_out, W_out]

        sampled = sampled.squeeze(2).permute(0, 2, 3, 1)  # [1, H_out, W_out, C]
        sampled = self.channel_proj(sampled).permute(0, 3, 1, 2)  # [1, C, H_out, W_out]
        return sampled

    def full_image(self, t, H=None, W=None):
        pred = self.forward(t, target_H=H, target_W=W)
        return pred.contiguous()


def tucker_construct(UT, UC, UH, UW, G, target_H=None, target_W=None):
    UT = UT.contiguous()
    UC = UC.contiguous()
    UH = UH.contiguous()
    UW = UW.contiguous()
    G = G.contiguous()

    def _col_norm(M, eps=1e-8):
        # column-wise L2 normalization
        if torch.is_complex(M):
            # ||col||^2 = sum(|z|^2) = sum(real^2 + imag^2)
            norms_sq = (M.real**2 + M.imag**2).sum(dim=0, keepdim=True)
            norms = torch.sqrt(norms_sq + eps)
        else:
            norms = M.norm(dim=0, keepdim=True) + eps
        return M / norms

    rH, rW, rC, rT = G.shape
    T, C, H, W = UT.shape[0], UC.shape[0], UH.shape[0], UW.shape[0]

    assert UT.shape[1] == rT and UC.shape[1] == rC and UH.shape[1] == rH and UW.shape[1] == rW
    if target_H is not None and target_H != H:
        UH = UH.T.unsqueeze(0)  # [1, rH, H]
        if UH.dtype == torch.complex64 or UH.dtype == torch.complex128:
            UH_real = F.interpolate(UH.real, size=target_H, mode='linear', align_corners=False)
            UH_imag = F.interpolate(UH.imag, size=target_H, mode='linear', align_corners=False)
            UH = torch.complex(UH_real, UH_imag)
        else:
            UH = F.interpolate(UH, size=target_H, mode='linear', align_corners=False)
        UH = UH.squeeze(0).T      # [H, rH]
        H = target_H
    if target_W is not None and target_W != W:
        UW = UW.T.unsqueeze(0)  # [1, rW, W]
        if UW.dtype == torch.complex64 or UW.dtype == torch.complex128:
            UW_real = F.interpolate(UW.real, size=target_W, mode='linear', align_corners=False)
            UW_imag = F.interpolate(UW.imag, size=target_W, mode='linear', align_corners=False)
            UW = torch.complex(UW_real, UW_imag)
        else:
            UW = F.interpolate(UW, size=target_W, mode='linear', align_corners=False)
        UW = UW.squeeze(0).T      # [W, rW]
        W = target_W
    
    # UT = _col_norm(UT)
    UC = _col_norm(UC)
    UH = _col_norm(UH)
    UW = _col_norm(UW)

    G_flat  = G.view(-1, rT)
    Gt_flat = (G_flat @ UT.T)
    Gt      = Gt_flat.view(rH, rW, rC, T).permute(0, 1, 3, 2)

    temp = (Gt @ UC.T).permute(2, 3, 1, 0)  # [T, C, rW, rH]
    temp = (temp @ UH.T).permute(0, 1, 3, 2)  # [T, C, H, rW]
    X = (temp @ UW.T).contiguous()  # [T, C, H, W]
    X = X.permute(1, 0, 2, 3).reshape(1, T * C, H, W)  # [1, T*C, H, W]
    
    return X


class BasicUpres(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, k, device='cuda'):
        super().__init__()
        half_k = k // 2
        self.k = k

        self.upres = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, dilation=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=3, dilation=3),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, dilation=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, (out_channels * (k**2)), kernel_size=1),
            nn.PixelShuffle(upscale_factor=k),
        ).to(device)
        print("BasicUpres params:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        upres_kernel = self.upres(x)
        return upres_kernel


class FlatConvRefine(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, device='cuda'):
        super().__init__()
        # Add a depthwise conv to mix channels minimally, then a LayerNorm for channel-wise normalization
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        ).to(device)

        print("FlatConvRefine params:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        refined = self.refine(x)
        return refined


class NikaBlock(nn.Module):
    def __init__(self, target_shape, tucker_ranks, grid_res, conv_hidden, out_channels, device):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.tucker_features = RealTucker(
            target_shape=target_shape,
            ranks=tucker_ranks,
            density=1,
            device=device,
        ).to(device)
        self.tucker_features.log_stats()

        self.complex_tucker = ComplexTucker(
            target_shape=target_shape,
            ranks=tucker_ranks,
            density=1,
            device=device,
        ).to(device)
        self.complex_tucker.log_stats()

        # self.tucker_refine = FlatConvRefine(
        #     in_channels=self.C * 3,
        #     out_channels=self.C,
        #     hidden=conv_hidden,
        #     device=device,
        # )

        self.grid_features = FeatureGrid(
            target_shape=target_shape,
            grid_res=grid_res,
            device=device,
        ).to(device)

        self.upres = BasicUpres(
            in_channels=2 * self.C,
            out_channels=out_channels,
            hidden=conv_hidden,
            k=4,
            device=device,
        )

    def forward(self, t):
        grid_out = self.grid_features.full_image(t, H=self.H, W=self.W)
        tucker_base = self.complex_tucker.full_image(t, t_range=1, H=self.H, W=self.W)
        tucker_res = self.tucker_features.full_image(t, t_range=1, H=self.H, W=self.W)
        tucker_out = tucker_base + tucker_res
        # tucker_out = self.tucker_refine(tucker_out)
        grid_out = torch.cat([grid_out, tucker_out], dim=1)
        refined = self.upres(grid_out)
        return refined


class FullNika(nn.Module):

    def __init__(self, feature_ranks, tucker_ranks, device='cuda'):
        super().__init__()
        self.nika_base = NikaBlock(
            target_shape=[3, 270, 480, 600],
            tucker_ranks=tucker_ranks,
            grid_res=feature_ranks,
            conv_hidden=128,
            out_channels=3,
            device=device,
        )

    def forward(self, t):
        prediction = self.nika_base(t)
        return prediction
            
    def test_images(self, output_dir):
        with torch.no_grad():
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            rand_vals = torch.linspace(0.0, 1.0, steps=10)
            for t in rand_vals:
                img = self.forward(t)
                save_image(img, os.path.join(output_dir, f"test_image_{t.item():.2f}.png"))


def feature_test(vid, device):
    downres_vid = vid[:, :, ::4, ::4]
    # downres_vid = vid
    print(f"Downres vid shape: {downres_vid.shape}")

    dense_tucker_ranks = [2, 80, 100, 60]
    smaller_tucker_ranks = [2, 60, 80, 40]
    # sparse_tucker_ranks = [2, 210, 240, 80]

    model = FullNika(
        feature_ranks=[2, 45, 60, 200],
        tucker_ranks=smaller_tucker_ranks,
        device=device
    )
    model.train()

    opt = SOAP(model.parameters(), lr=1e-2)

    for epoch in range(2000):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        start_time = time.time()
        for t in range(vid.shape[0]):
            gt = vid[t:t+1].to(torch.float32) / 255.0
            norm_t = t / (vid.shape[0] - 1)
            prediction = model(norm_t)
            # pred_upres = F.interpolate(prediction, size=(vid.shape[2], vid.shape[3]), mode='bilinear', align_corners=False)
            mse = F.mse_loss(prediction, gt)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            frame_loss = (-psnr).mean() / vid.shape[0]
            frame_loss.backward()
            loss += frame_loss
        average_frame_time = (time.time() - start_time) / vid.shape[0]
        print(f"Epoch {epoch} loss: {loss.item():.4f}, time: {average_frame_time:.5f}s")
        opt.step()

        if epoch % 25 == 0:
            batch_psnr = loss.item() * -1.0

            print(f"Epoch {epoch}: Tucker PSNR: {batch_psnr:.2f}")
            for t in torch.linspace(0.0, 1.0, steps=10):
                with torch.no_grad():
                    test_img = model(t.item())
                    os.makedirs("out_feature_test", exist_ok=True)
                    save_image(test_img, f"out_feature_test/test_image_epoch{epoch}_t{t.item():.2f}.png")
            # model.test_images("out_feature_test")


if __name__ == "__main__":
    device = "cuda:1"
    vid = load_video_frames("static/benchmarks/uvg/bosphorus", device, max_frames=600, dtype=torch.uint8, normalize=False)
    feature_test(vid, device=device)
    # baseline_video_test(vid, k=4, ranks=[2, 180, 194, 42], mlp_hidden=256, device=device)
