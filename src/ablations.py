import os
import torch
import torch.nn as nn

from load_data import load_video_frames
from nika import RealTucker, ComplexTucker, FeatureGrid, BasicUpres
from soap import SOAP
from configs import REFERENCES


class RealNika(nn.Module):
    def __init__(self, target_shape, k, real_tucker_ranks, grid_ranks, conv_hidden, out_channels, device):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.H = int(self.H // k); self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]
        self.real_tucker = RealTucker(
            target_shape=self.internal_shape,
            ranks=real_tucker_ranks,
            device=device,
        ).to(device)
        self.feature_grid = FeatureGrid(
            target_shape=self.internal_shape,
            grid_res=grid_ranks,
            device=device,
        ).to(device)
        
        self.upres = BasicUpres(
            in_channels=2 * self.C,
            out_channels=out_channels,
            hidden=conv_hidden,
            k=k,
            device=device,
        ).to(device)

        self.groupnorm = nn.GroupNorm(num_groups=2, num_channels=2 * self.C).to(device)
        self.log_stats()

    def log_stats(self):
        feature_grid_params = sum(p.numel() for p in self.feature_grid.parameters())
        real_tucker_params = sum(p.numel() for p in self.real_tucker.parameters())
        upres_params = sum(p.numel() for p in self.upres.parameters())
        total_params = feature_grid_params + real_tucker_params + upres_params
        print(f"NikaBlock parameters:")
        print(f"  Feature Grid:    {feature_grid_params / 1e6:.3f}M")
        print(f"  Real Tucker:     {real_tucker_params / 1e6:.3f}M")
        print(f"  Upsampling CNN:  {upres_params / 1e6:.3f}M")
        print(f"  Total:           {total_params / 1e6:.3f}M")

    def forward(self, t):
        if type(t) is not torch.Tensor:
            t = torch.tensor([t], device=self.real_tucker.grid.device, dtype=torch.int64)
        feature_grid_base = self.feature_grid(t)
        real_tucker_out = self.real_tucker(t)

        base_input = torch.cat([feature_grid_base, real_tucker_out], dim=1)
        normed_input = self.groupnorm(base_input)
        refined = self.upres(normed_input)
        return refined


class TuckerNika(nn.Module):
    def __init__(self, target_shape, k, real_tucker_ranks, complex_tucker_ranks, conv_hidden, out_channels, device):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.H = int(self.H // k); self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]
        self.real_tucker = RealTucker(
            target_shape=self.internal_shape,
            ranks=real_tucker_ranks,
            device=device,
        ).to(device)

        self.complex_tucker = ComplexTucker(
            target_shape=self.internal_shape,
            ranks=complex_tucker_ranks,
            device=device,
        ).to(device)

        self.upres = BasicUpres(
            in_channels=2 * self.C,
            out_channels=out_channels,
            hidden=conv_hidden,
            k=k,
            device=device,
        ).to(device)

        self.groupnorm = nn.GroupNorm(num_groups=2, num_channels=2 * self.C).to(device)
        self.log_stats()

    def log_stats(self):
        real_tucker_params = sum(p.numel() for p in self.real_tucker.parameters())
        complex_tucker_params = sum(p.numel() for p in self.complex_tucker.parameters())
        upres_params = sum(p.numel() for p in self.upres.parameters())
        total_params = real_tucker_params + complex_tucker_params + upres_params
        print(f"NikaBlock parameters:")
        print(f"  Real Tucker:     {real_tucker_params / 1e6:.3f}M")
        print(f"  Complex Tucker:  {complex_tucker_params / 1e6:.3f}M")
        print(f"  Upsampling CNN:  {upres_params / 1e6:.3f}M")
        print(f"  Total:           {total_params / 1e6:.3f}M")

    def forward(self, t):
        if type(t) is not torch.Tensor:
            t = torch.tensor([t], device=self.real_tucker.grid.device, dtype=torch.int64)
        real_tucker_out = self.real_tucker(t)
        complex_tucker_out = self.complex_tucker(t)
        base_input = torch.cat([real_tucker_out, complex_tucker_out], dim=1)
        normed_input = self.groupnorm(base_input)
        refined = self.upres(normed_input)
        return refined


class WeirdNika(nn.Module):
    def __init__(self, target_shape, k, complex_tucker_ranks, conv_hidden, grid_ranks, out_channels, device):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.H = int(self.H // k); self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]
        self.feature_grid = FeatureGrid(
            target_shape=self.internal_shape,
            grid_res=grid_ranks,
            device=device,
        ).to(device)

        self.complex_tucker = ComplexTucker(
            target_shape=self.internal_shape,
            ranks=complex_tucker_ranks,
            device=device,
        ).to(device)

        self.upres = BasicUpres(
            in_channels=2 * self.C,
            out_channels=out_channels,
            hidden=conv_hidden,
            k=k,
            device=device,
        ).to(device)

        self.groupnorm = nn.GroupNorm(num_groups=2, num_channels=2 * self.C).to(device)
        self.log_stats()

    def log_stats(self):
        feature_grid_params = sum(p.numel() for p in self.feature_grid.parameters())
        complex_tucker_params = sum(p.numel() for p in self.complex_tucker.parameters())
        upres_params = sum(p.numel() for p in self.upres.parameters())
        total_params = feature_grid_params + complex_tucker_params + upres_params
        print(f"NikaBlock parameters:")
        print(f"  Feature Grid:    {feature_grid_params / 1e6:.3f}M")
        print(f"  Complex Tucker:  {complex_tucker_params / 1e6:.3f}M")
        print(f"  Upsampling CNN:  {upres_params / 1e6:.3f}M")
        print(f"  Total:           {total_params / 1e6:.3f}M")

    def forward(self, t):
        if type(t) is not torch.Tensor:
            t = torch.tensor([t], device=self.feature_grid.grid.device, dtype=torch.int64)
        feature_grid_out = self.feature_grid(t)
        complex_tucker_out = self.complex_tucker(t)
        base_input = torch.cat([feature_grid_out, complex_tucker_out], dim=1)
        normed_input = self.groupnorm(base_input)
        refined = self.upres(normed_input)
        return refined


class NoConvNika(nn.Module):
    def __init__(self, target_shape, k, real_tucker_ranks, complex_tucker_ranks, grid_ranks, out_channels, device):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.H = int(self.H // k); self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]
        self.feature_grid = FeatureGrid(
            target_shape=self.internal_shape,
            grid_res=grid_ranks,
            device=device,
        ).to(device)

        self.real_tucker = RealTucker(
            target_shape=self.internal_shape,
            ranks=real_tucker_ranks,
            device=device,
        ).to(device)

        self.complex_tucker = ComplexTucker(
            target_shape=self.internal_shape,
            ranks=complex_tucker_ranks,
            device=device,
        ).to(device)

        self.upres = nn.Sequential(
            nn.Conv2d(3 * self.C, 3 * k**2, kernel_size=1),
            nn.PixelShuffle(upscale_factor=k),
        ).to(device)
        self.log_stats()

    def log_stats(self):
        feature_grid_params = sum(p.numel() for p in self.feature_grid.parameters())
        real_tucker_params = sum(p.numel() for p in self.real_tucker.parameters())
        complex_tucker_params = sum(p.numel() for p in self.complex_tucker.parameters())
        upres_params = sum(p.numel() for p in self.upres.parameters())
        total_params = feature_grid_params + real_tucker_params + complex_tucker_params + upres_params
        print(f"NikaBlock parameters:")
        print(f"  Feature Grid:    {feature_grid_params / 1e6:.3f}M")
        print(f"  Real Tucker:     {real_tucker_params / 1e6:.3f}M")
        print(f"  Complex Tucker:  {complex_tucker_params / 1e6:.3f}M")
        print(f"  Upsampling CNN:  {upres_params / 1e6:.3f}M")
        print(f"  Total:           {total_params / 1e6:.3f}M")

    def forward(self, t):
        if type(t) is not torch.Tensor:
            t = torch.tensor([t], device=self.feature_grid.grid.device, dtype=torch.int64)
        feature_grid_out = self.feature_grid(t)
        real_tucker_out = self.real_tucker(t)
        complex_tucker_out = self.complex_tucker(t)
        base_input = torch.cat([feature_grid_out, real_tucker_out, complex_tucker_out], dim=1)
        refined = self.upres(base_input)
        return refined


def run_ablation(config, dataset_dir, device):
    # names = ["honey", "jockey", "ready"]
    names = ["ready", "shake", "yacht"]
    batch_size = 5

    config_kwargs = {**REFERENCES[config]}
    for name in names:
        data = load_video_frames(dataset_dir + f"/{name}", device=device, max_frames=600, normalize=False)
        print(f"Running ablation for video: {name}, config: {config}")
        epochs = 1500
        model = WeirdNika(
            target_shape=[4, data.shape[2], data.shape[3], data.shape[0]],
            k=4,
            **config_kwargs,
            out_channels=3,
            device=device
        )
        opt = SOAP(model.parameters(), lr=1e-2)

        best_psnr = float('-inf')
        best_epoch = -1
        for epoch in range(epochs):
            opt.zero_grad(set_to_none=True)
            total_psnr = 0.0
            total_frames = 0
            num_batches = (data.shape[0] + batch_size - 1) // batch_size
            for t in range(num_batches):
                min_t = t * batch_size
                max_t = min((t + 1) * batch_size, data.shape[0])
                batch_gt = data[min_t:max_t].to(torch.float32) / 255.0

                t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
                prediction = model(t_batch)
                mse = torch.nn.functional.mse_loss(prediction, batch_gt)
                psnr = -10.0 * torch.log10(mse + 1e-8)
                # Accumulate PSNR for each frame in batch
                batch_size_actual = batch_gt.shape[0]
                total_psnr += psnr.item() * batch_size_actual
                total_frames += batch_size_actual
                # Backward pass on negative PSNR (maximize PSNR)
                loss = -psnr
                loss.backward()
            epoch_psnr = total_psnr / total_frames
            print(f"[{name}] Epoch {epoch} PSNR: {epoch_psnr:.4f}")

            if epoch_psnr > best_psnr and (epoch - best_epoch >= 10 or best_epoch == -1):
                best_psnr = epoch_psnr
                best_epoch = epoch
                model_path = f"models/{config}-{name}-epoch{epoch}-psnr{best_psnr:.2f}.torch"
                torch.save(model.state_dict(), model_path)
                os.sync()
                print(f"New best model saved at epoch {epoch} with PSNR: {best_psnr:.2f}")

            opt.step()

        print(f"[{name}] Best PSNR achieved: {best_psnr:.2f} at epoch {best_epoch}")

if __name__ == "__main__":
    device="cuda:0"
    run_ablation("weird-nika", "static/benchmarks/uvg", device)
