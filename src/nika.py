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
        paths = all_paths[:max_frames]
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
        vid_cpu[t].copy_(lin)

    return vid_cpu.to(device=device, dtype=dtype, non_blocking=True)


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
        if self.chunked:
            chunk_idx = target // self.max_chunk_size
            rel_idx = target % self.max_chunk_size
            if self.is_complex:
                r_chunks = [self.real_chunks[c][r:r+1] for c, r in zip(chunk_idx, rel_idx)]
                i_chunks = [self.imag_chunks[c][r:r+1] for c, r in zip(chunk_idx, rel_idx)]
                U_target = torch.complex(torch.cat(r_chunks, dim=0), torch.cat(i_chunks, dim=0))
            else:
                chunks = [self.chunks[c][r:r+1] for c, r in zip(chunk_idx, rel_idx)]
                U_target = torch.cat(chunks, dim=0)
        else:
            if self.is_complex:
                U_target = torch.complex(
                    self.U_real[target],
                    self.U_imag[target],
                )
            else:
                U_target = self.U[target]
        return U_target


class RealTucker(nn.Module):
    def __init__(self, target_shape, ranks, device='cuda'):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.rC, self.rH, self.rW, self.rT = ranks

        self.UH = TuckerFactor(self.H, self.rH, is_complex=False, device=device)
        self.UW = TuckerFactor(self.W, self.rW, is_complex=False, device=device)
        self.UC = TuckerFactor(self.C, self.rC, is_complex=False, device=device)
        self.UT = TuckerFactor(self.T, self.rT, is_complex=False, device=device)

        self.G = nn.Parameter(torch.randn(self.rH, self.rW, self.rC, self.rT, device=device) * 1e-2)

    def forward(self, t):
        UT = self.UT.get(t)
        UC = self.UC()
        UH = self.UH()
        UW = self.UW()
        return tucker_construct(UT, UC, UH, UW, self.G).contiguous()


class ComplexTucker(RealTucker):

    def __init__(self, target_shape, ranks, device='cuda'):
        super().__init__(target_shape, ranks, device=device)
        self.UH = TuckerFactor(self.H, self.rH, is_complex=True, device=device)
        self.UW = TuckerFactor(self.W, self.rW, is_complex=True, device=device)
        self.UC = TuckerFactor(self.C, self.rC, is_complex=True, device=device)
        self.UT = TuckerFactor(self.T, self.rT, is_complex=True, device=device)

        self.G = None  # override parent
        self.G_real = nn.Parameter(torch.randn(self.rH, self.rW, self.rC, self.rT, device=device) * 1e-2)
        self.G_imag = nn.Parameter(torch.zeros(self.rH, self.rW, self.rC, self.rT, device=device))

    def forward(self, t):
        UH = self.UH()
        UW = self.UW()
        UC = self.UC()
        UT = self.UT.get(t)
        G = torch.complex(self.G_real, self.G_imag)
        construct = tucker_construct(UT, UC, UH, UW, G)
        base = hermitian_project_shifted(construct)
        real_tucker = torch.fft.ifft2(base, norm='ortho').real
        return real_tucker.contiguous()


class FeatureGrid(nn.Module):
    def __init__(self, target_shape, grid_res, zero_init=False, device="cuda"):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.grid_c = grid_res[0]
        self.grid_h = grid_res[1]
        self.grid_w = grid_res[2]
        self.grid_t = grid_res[3]

        # Initialize parameters to log of uniform random values in (0, 1)
        self.grid = nn.Parameter(torch.log(torch.rand(self.grid_c, self.grid_h, self.grid_w, self.grid_t, device=device) + 1e-8))
        self.channel_proj = nn.Linear(self.grid_c, self.C, bias=True).to(device)

    def forward(self, t, target_H=None, target_W=None):
        device = self.grid.device
        H_out = target_H if target_H is not None else self.H
        W_out = target_W if target_W is not None else self.W

        B = t.shape[0]
        t_norm = (2.0 * t) - 1.0
        y_lin = torch.linspace(0, self.H - 1, steps=H_out, device=device)
        x_lin = torch.linspace(0, self.W - 1, steps=W_out, device=device)
        y_norm = 2.0 * (y_lin / (self.H - 1)) - 1.0
        x_norm = 2.0 * (x_lin / (self.W - 1)) - 1.0

        y_lin = torch.linspace(0, self.H - 1, steps=H_out, device=device)
        x_lin = torch.linspace(0, self.W - 1, steps=W_out, device=device)
        y_norm = 2.0 * (y_lin / (self.H - 1)) - 1.0
        x_norm = 2.0 * (x_lin / (self.W - 1)) - 1.0

        y, x = torch.meshgrid(y_norm, x_norm, indexing='ij')  # [H_out, W_out]
        y = y.expand(B, H_out, W_out)
        x = x.expand(B, H_out, W_out)
        t_grid = t_norm.view(B, 1, 1).expand(B, H_out, W_out)
        sample_grid = torch.stack((x, y, t_grid), dim=-1)  # [B, H_out, W_out, 3]
        sample_grid = sample_grid.unsqueeze(1)  # [B, 1, H_out, W_out, 3]

        grid_5d = self.grid.permute(0, 3, 1, 2).unsqueeze(0)  # [1, C, T_g, H_g, W_g]
        grid_5d = grid_5d.expand(B, -1, -1, -1, -1)  # [B, C, T_g, H_g, W_g]

        sampled = F.grid_sample(
            grid_5d,               # [B, C, T_g, H_g, W_g]
            sample_grid,           # [B, 1, H_out, W_out, 3]
            mode='bilinear',
            align_corners=True,
            padding_mode='border',
        )  # → [B, C, 1, H_out, W_out]

        sampled = sampled.squeeze(2).permute(0, 2, 3, 1)  # [B, H_out, W_out, C]
        sampled = self.channel_proj(sampled).permute(0, 3, 1, 2)  # [B, C, H_out, W_out]
        return sampled.contiguous()


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

    X = torch.einsum('ijkl,tl,ck,hi,wj->tchw', G, UT, UC, UH, UW)
    return X


def cp_construct(UT, UC, UH, UW):
    UT = UT.contiguous()
    UC = UC.contiguous()
    UH = UH.contiguous()
    UW = UW.contiguous()

    r = UT.shape[1]
    T, C, H, W = UT.shape[0], UC.shape[0], UH.shape[0], UW.shape[0]

    assert UC.shape[1] == r and UH.shape[1] == r and UW.shape[1] == r

    X = torch.zeros((T, C, H, W), device=UT.device, dtype=UT.dtype)

    for i in range(r):
        outer_product = torch.einsum('t,c,h,w->tc hw', UT[:, i], UC[:, i], UH[:, i], UW[:, i])
        X += outer_product.reshape(T, C, H, W)

    return X

class FactorizedCore(nn.Module):
    def __init__(self, target_shape, rank, device='cuda'):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.r = rank

        self.UH = TuckerFactor(self.H, self.r, is_complex=False, device=device)
        self.UW = TuckerFactor(self.W, self.r, is_complex=False, device=device)
        self.UC = TuckerFactor(self.C, self.r, is_complex=False, device=device)
        self.UT = TuckerFactor(self.T, self.r, is_complex=False, device=device)

    def forward(self, t):
        UT = self.UT.get(t)
        UC = self.UC()
        UH = self.UH()
        UW = self.UW()
        return cp_construct(UT, UC, UH, UW).contiguous()

class BasicUpres(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, k, blocks, device='cuda'):
        super().__init__()
        half_k = k // 2
        self.k = k

        layers = [
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        for _ in range(blocks):
            layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=k),
        ]
        self.upres = nn.Sequential(*layers).to(device)

        # xavier init
        for m in self.upres.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        self.inputs = x
        upres_kernel = self.upres(x)
        return upres_kernel


class NikaBlock(nn.Module):
    def __init__(self, target_shape, k, real_tucker_ranks, complex_tucker_ranks, grid_ranks, conv_hidden, upres_blocks, out_channels, device):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.H = int(self.H // k); self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]
        self.real_tucker = RealTucker(
            target_shape=self.internal_shape,
            ranks=real_tucker_ranks,
            device=device,
        ).to(device)
        torch.compile(self.real_tucker)

        self.complex_tucker = ComplexTucker(
            target_shape=self.internal_shape,
            ranks=complex_tucker_ranks,
            device=device,
        ).to(device)
        torch.compile(self.complex_tucker)

        self.grid_features = FeatureGrid(
            target_shape=self.internal_shape,
            grid_res=grid_ranks,
            device=device,
        ).to(device)
        torch.compile(self.grid_features)

        self.upres = BasicUpres(
            in_channels=3 * self.C,
            out_channels=out_channels,
            hidden=conv_hidden,
            k=k,
            blocks=upres_blocks,
            device=device,
        ).to(device)
        torch.compile(self.upres)

        self.groupnorm = nn.GroupNorm(num_groups=3, num_channels=3 * self.C).to(device)
        torch.compile(self.groupnorm)
        self.log_stats()

    def log_stats(self):
        real_tucker_params = sum(p.numel() for p in self.real_tucker.parameters())
        complex_tucker_params = sum(p.numel() for p in self.complex_tucker.parameters())
        grid_params = sum(p.numel() for p in self.grid_features.parameters())
        upres_params = sum(p.numel() for p in self.upres.parameters())
        total_params = real_tucker_params + complex_tucker_params + grid_params + upres_params
        print(f"NikaBlock parameters:")
        print(f"  Real Tucker:     {real_tucker_params / 1e6:.3f}M")
        print(f"  Complex Tucker:  {complex_tucker_params / 1e6:.3f}M")
        print(f"  Feature Grid:    {grid_params / 1e6:.3f}M")
        print(f"  Upsampling CNN:  {upres_params / 1e6:.3f}M")
        print(f"  Total:           {total_params / 1e6:.3f}M")

    def forward(self, t):
        grid_out = self.grid_features(t / (self.T - 1))
        real_tucker_out = self.real_tucker(t)
        complex_tucker_out = self.complex_tucker(t)
        normed_input = torch.cat([grid_out, real_tucker_out, complex_tucker_out], dim=1)
        core_input = self.groupnorm(normed_input)
        refined = self.upres(core_input)
        return refined

    def test_images(self, output_dir):
        self.eval()
        with torch.no_grad():
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            rand_vals = torch.linspace(0, self.T - 1, steps=10, dtype=torch.int64, device=self.grid_features.grid.device)
            torch.cuda.synchronize()
            start_time = time.time()
            imgs = self.forward(rand_vals)
            torch.cuda.synchronize()
            average_frame_time = (time.time() - start_time) / rand_vals.shape[0]
            print(f"Average inference time per frame: {average_frame_time:.5f}s")
            print(f"FPS: {1.0 / average_frame_time:.2f}")
            for i in range(imgs.shape[0]):
                img = imgs[i].clamp(0.0, 1.0)
                save_image(img, f"{output_dir}/frame_{i:04d}.png")


def feature_test(vid, device):
    batch_size = 10
    # reference sizes
    # 3.3M config
    # small_grid_ranks = [2, 60, 70, 120]  # 1M params
    # small_real_tucker = [2, 80, 80, 80]  # 1.1M params
    # small_complex_tucker = [2, 60, 60, 60]  # 1M params
    # small_hidden = 150  # 0.1M params
    # small_upres_blocks = 1
    small_grid_ranks = [2, 60, 70, 120]
    small_real_tucker = [2, 80, 80, 80]
    small_complex_tucker = [2, 60, 60, 60]
    small_hidden = 150
    small_upres_blocks = 1

    # 6M config
    # medium_grid_ranks = [3, 70, 80, 130]
    # medium_real_tucker = [3, 85, 85, 85]
    # medium_complex_tucker = [3, 65, 65, 65]
    # medium_hidden = 200

    medium_grid_ranks = [2, 80, 90, 150]
    medium_real_tucker = [2, 100, 100, 100]
    medium_complex_tucker = [2, 70, 70, 70]
    medium_hidden = 200
    medium_upres_blocks = 1

    model = NikaBlock(
        target_shape=[3, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        real_tucker_ranks=medium_real_tucker,
        complex_tucker_ranks=medium_complex_tucker,
        grid_ranks=medium_grid_ranks,
        conv_hidden=medium_hidden,
        upres_blocks=medium_upres_blocks,
        out_channels=3,
        device=device,
    )

    opt = SOAP(list(model.parameters()), lr=1e-2)

    for epoch in range(2000):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        start_time = time.time()
        for t in range(vid.shape[0] // batch_size):
            min_t = t * batch_size; max_t = (t + 1) * batch_size
            batch_gt = vid[min_t:max_t].to(torch.float32) / 255.0
            t_batch = torch.linspace(min_t, max_t - 1, steps=(max_t - min_t), dtype=torch.int64, device=device)
            prediction = model(t_batch)
            mse = F.mse_loss(prediction, batch_gt)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            frame_loss = (-psnr).mean() / (vid.shape[0] // batch_size)
            frame_loss.backward()
            loss += frame_loss
        average_frame_time = (time.time() - start_time) / vid.shape[0]
        print(f"Epoch {epoch} loss: {loss.item():.4f}, time: {average_frame_time:.5f}s")
        opt.step()

        if epoch % 25 == 0:
            batch_psnr = loss.item() * -1.0
            print(f"Epoch {epoch}: Tucker PSNR: {batch_psnr:.2f}")
            model.test_images("out_feature_test")

    torch.save(model.state_dict(), "models/beauty-model.torch")


def explain(vid, device):
    from torch.autograd import grad
    batch_size = 1

    # reference sizes
    # 3.3M config
    small_grid_ranks = [2, 60, 70, 120]  # 1M params
    small_real_tucker = [2, 80, 80, 80]  # 1.1M params
    small_complex_tucker = [2, 60, 60, 60]  # 1M params
    small_hidden = 150  # 0.1M params

    model = NikaBlock(
        target_shape=[3, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        real_tucker_ranks=small_real_tucker,
        complex_tucker_ranks=small_complex_tucker,
        grid_ranks=small_grid_ranks,
        conv_hidden=small_hidden,
        out_channels=3,
        device=device,
    )
    model.load_state_dict(torch.load("models/beauty-model.torch"))
    model.eval()

    opt = SOAP(list(model.parameters()), lr=1e-2)

    opt.zero_grad(set_to_none=True)
    loss = 0.0
    start_time = time.time()

    def rescale(img):
        rescaled = (img - img.min()) / (img.max() - img.min())
        return rescaled

    for t in range(vid.shape[0] // batch_size):
        min_t = t * batch_size; max_t = (t + 1) * batch_size
        batch_gt = vid[min_t:max_t].to(torch.float32) / 255.0
        t_batch = torch.linspace(min_t, max_t - 1, steps=(max_t - min_t), dtype=torch.int64, device=device)
        prediction = model(t_batch)
        pixel_sum = prediction.sum()
        upres_input_grad = grad(pixel_sum, model.upres.inputs)[0]
        contributions = upres_input_grad.abs().mean(dim=(2, 3))
        print(contributions)
        absed = []
        for i in range(9):
            save_image(rescale(upres_input_grad[0,i,...]), f"visuals/frame0_channel{i}.png")
            abs_i = rescale(upres_input_grad[0,i,...]).abs()
            save_image(abs_i, f"visuals/frame0_channel{i}_abs.png")
            absed.append(abs_i)

        intensities = []
        for start in range(0, 9, 3):
            combined = torch.stack(absed[start:start+3], axis=0)
            intensity = combined.norm(dim=0)
            intensities.append(intensity)
            intensity = intensity / intensity.max()
            names = {
                0: 'real',
                3: 'imaginary',
                6: 'feature_grid'
            }
            save_image(combined, f"visuals/{names[start]}_independent_norm.png")
            save_image(intensity, f"visuals/{names[start]}_intensity.png")

        save_image(rescale(torch.stack(intensities, axis=0)), "visuals/merge_all_the_things.png")
        save_image(rescale(upres_input_grad[0,:3,...]), "visuals/real_branch_frame0.png")
        save_image(rescale(upres_input_grad[0,3:6,...]), "visuals/imaginary_branch_frame0.png")
        save_image(rescale(upres_input_grad[0,6:,...]), "visuals/feature_grid_branch_frame0.png")
        save_image(rescale(upres_input_grad[0,:3,...]).abs(), "visuals/real_branch_abs_frame0.png")
        save_image(rescale(upres_input_grad[0,3:6,...]).abs(), "visuals/imaginary_branch_abs_frame0.png")
        save_image(rescale(upres_input_grad[0,6:,...]).abs(), "visuals/feature_grid_branch_abs_frame0.png")
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    device = "cuda:1"
    vid = load_video_frames("static/benchmarks/uvg/jockey", device, max_frames=600, dtype=torch.uint8, normalize=False)
    # explain(vid, device=device)
    feature_test(vid, device=device)
