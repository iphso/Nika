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
from configs import REFERENCES


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
        return U[target]


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

        self.grid = nn.Parameter(torch.randn(self.grid_c, self.grid_h, self.grid_w, self.grid_t, device=device) * 1e-2)
        self.channel_proj = nn.Linear(self.grid_c, self.C, bias=True).to(device)
        nn.init.normal_(self.channel_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.channel_proj.bias)
        self.register_buffer("_xy_base", None, persistent=False)
        self._grid_5d_view = None
        self._generate_xy_base()

    def _generate_xy_base(self):
        device = self.grid.device
        y_lin = torch.arange(0, self.H, device=device)
        x_lin = torch.arange(0, self.W, device=device)
        y_norm = 2.0 * (y_lin / (self.H - 1)) - 1.0
        x_norm = 2.0 * (x_lin / (self.W - 1)) - 1.0
        y, x = torch.meshgrid(y_norm, x_norm, indexing='ij')  # [H, W]
        self._xy_base = torch.stack((x, y), dim=-1)  # [H, W, 2]
    
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
            align_corners=True,
            padding_mode='border',
        )  # → [B, C, 1, H_out, W_out]

        sampled = sampled.squeeze(2).permute(0, 2, 3, 1)  # [B, H_out, W_out, C]
        result = self.channel_proj(sampled).permute(0, 3, 1, 2)  # [B, C, H_out, W_out]
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

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=1),
            nn.PixelShuffle(upscale_factor=k),
        ])
        self.upres = nn.Sequential(*self.layers).to(device)

        #kaiming init
        for m in self.upres.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, log_times=False):
        if not log_times:
            return self.upres(x)
        times = []
        inp = x
        for idx, layer in enumerate(self.layers):
            torch.cuda.synchronize()
            start = time.time()
            inp = layer(inp)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append((type(layer).__name__, elapsed))
        for i, (name, t) in enumerate(times):
            print(f"Layer {i}: {name} took {t*1000:.3f} ms")
        return inp


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
        if type(t) is not torch.Tensor:
            t = torch.tensor([t], device=self.grid_features.grid.device, dtype=torch.int64)
        grid_out = self.grid_features(t / (self.T - 1))
        real_tucker_out = self.real_tucker(t)
        complex_tucker_out = self.complex_tucker(t)
        base_input = torch.cat([grid_out, real_tucker_out, complex_tucker_out], dim=1)
        normed_input = self.groupnorm(base_input)
        refined = self.upres(normed_input)
        return refined

    def test_images(self, output_dir):
        # self.eval()
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


def feature_test(vid, name, config, device):
    batch_size = 10
    model_kwargs = REFERENCES[config]
    model = NikaBlock(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        **model_kwargs,
        out_channels=3,
        device=device,
    )

    opt = SOAP(list(model.parameters()), lr=1e-2)

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
            prediction = model(t_batch)
            mse = F.mse_loss(prediction, batch_gt)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            frame_loss = (-psnr).mean() / num_batches
            frame_loss.backward()
            loss += frame_loss
        average_frame_time = (time.time() - start_time) / vid.shape[0]
        epoch_psnr = -loss.item()
        print(f"Epoch {epoch} loss: {loss.item():.4f}, time: {average_frame_time:.5f}s, PSNR: {epoch_psnr:.2f}")

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
    device = "cuda:1"
    name = "bunny"
    vid = load_video_frames(f"static/benchmarks/{name}", device, max_frames=600, dtype=torch.uint8, normalize=False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # explain(vid, device=device)
    feature_test(vid, name, "xxs", device=device)
    # batch_profile(vid, device=device, batch_sizes=(1, 5, 10), iters=10, warmup=5)
