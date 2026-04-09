import os
import time
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.autograd import grad

import re
import glob
from load_data import load_video_frames
from nika import NikaBlock
from soap import SOAP
from configs import REFERENCES
import subprocess

def get_best_model(model_dir, vid_shape, vid_name, config, device):
    all_models = glob.glob(f"{model_dir}/{config}-{vid_name}-*.torch")
    if not all_models:
        raise ValueError(f"No models found for {vid_name} with config {config}")

    # Sort models by PSNR (extracting the PSNR value from the filename)
    def extract_psnr(filename):
        match = re.search(r'psnr([0-9]+(?:\.[0-9]+)?)', filename)
        if match:
            return float(match.group(1))
        raise ValueError(f"Could not extract PSNR from filename: {filename}")
    all_models.sort(key=extract_psnr)

    print(f"Best model for {vid_name} with config {config}: {all_models[-1]}")
    model = NikaBlock(
        target_shape=[4, vid_shape[2], vid_shape[3], vid_shape[0]],
        k=4,
        **REFERENCES[config],
        out_channels=3,
        device=device,
    )
    model_path = all_models[-1]
    # model_path = "models/ref_models/small-beauty-epoch1999-psnr33.36.torch"
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Handle torch.compile saved state_dicts with _orig_mod prefixes
        if any("_orig_mod" in k for k in state_dict.keys()):
            cleaned_state = {}
            for k, v in state_dict.items():
                cleaned_key = k.replace("._orig_mod", "")
                cleaned_state[cleaned_key] = v
            model.load_state_dict(cleaned_state)
        else:
            raise e
    return model


def benchmark_psnr(basedir, vid_name, config, device):
    vid = load_video_frames(f"{basedir}/{vid_name}", device, max_frames=600, dtype=torch.uint8, normalize=False)
    model = get_best_model(f"models/ref_models/", vid.shape, vid_name, config, device)

    core_image = model.grid_features.grid.data.cpu().numpy().copy()
    print(f"Core image shape: {core_image.shape}, value range: [{core_image.min()}, {core_image.max()}]")

    # Convert to tensor and make it a valid image: [C, H, W], normalized to [0, 1]
    core_t = torch.from_numpy(core_image).squeeze(-1).to(torch.float32)
    # Use first 3 channels for RGB visualization
    core_t = core_t[:3, ...]
    # Percentile-based normalization to avoid gray-looking images
    q_low = torch.quantile(core_t, 0.01)
    q_high = torch.quantile(core_t, 0.99)
    core_t = (core_t - q_low) / (q_high - q_low + 1e-8)
    core_t = core_t.clamp(0.0, 1.0)
    save_image(core_t, f"visuals/{vid_name}/{config}/core_image.png")
    os.makedirs(f"visuals/{vid_name}/{config}/preds", exist_ok=True)
    os.makedirs(f"visuals/{vid_name}/{config}/residual", exist_ok=True)
    model.eval()
    total_psnr = 0.0
    num_frames = vid.shape[0]

    batch_size = 6
    num_batches = (num_frames + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_idx in range(num_batches):
            min_t = batch_idx * batch_size
            max_t = min((batch_idx + 1) * batch_size, num_frames)
            batch_gt = vid[min_t:max_t].to(torch.float32) / 255.0
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            prediction = model(t_batch)
            residual = prediction - batch_gt
            residual_max = residual.max(); residual_min = residual.min()
            for i in range(prediction.shape[0]):
                save_image(prediction[i], f"visuals/{vid_name}/{config}/preds/pred_frame{min_t + i:03d}.png")
                # Map residual to [0, 1] by normalizing to its min/max per-frame
                res = residual[i]
                res_norm = torch.abs(res) * 5.0  # scale up for visibility
                save_image(res_norm, f"visuals/{vid_name}/{config}/residual/residual_frame{min_t + i:03d}.png")
                # FFT of residual: log-magnitude, centered, normalized per-frame
                try:
                    fft_res = torch.fft.fft2(res)
                    fft_mag = torch.abs(torch.fft.fftshift(fft_res, dim=(-2, -1)))
                    fft_log = torch.log1p(fft_mag)
                    fft_norm = fft_log / (fft_log.max() + 1e-8)
                    save_image(fft_norm, f"visuals/{vid_name}/{config}/residual_fft/residual_fft_frame{min_t + i:03d}.png")
                except Exception:
                    # If FFT/save fails, skip silently to avoid breaking benchmark
                    pass
                mse = F.mse_loss(prediction[i].clamp(0, 1), batch_gt[i])
                psnr = 10 * torch.log10(1 / (mse + 1e-8))
                total_psnr += psnr.item()
                if (min_t + i) % 100 == 0:
                    print(f"Processed frame {min_t + i}, PSNR: {psnr:.4f}")

    avg_psnr = total_psnr / num_frames
    print(f"Average PSNR: {avg_psnr:.4f}")

    # Timing run
    if "cuda" in device:
        torch.cuda.synchronize(device)
    start_time = time.time()
    with torch.no_grad():
        for batch_idx in range(num_batches):
            min_t = batch_idx * batch_size
            max_t = min((batch_idx + 1) * batch_size, num_frames)
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            _ = model(t_batch)
    if "cuda" in device:
        torch.cuda.synchronize(device)
    end_time = time.time()
    print(f"Timing run took {end_time - start_time:.4f} seconds")


def benchmark_fps_torch_events(
    basedir,
    vid_name,
    config,
    device,
    n_frames=300,
    batch_size=1,
    warmup_iters=10,
    repeats=50,
    profile=True,
    profile_dir="profiles",
    profile_name="fps_benchmark",
):
    if "cuda" not in device:
        raise ValueError("This benchmark uses CUDA events; please use a CUDA device.")

    # Determine model input shape using a single-frame probe (avoids loading full video)
    probe = load_video_frames(f"{basedir}/{vid_name}", device, max_frames=1, dtype=torch.uint8, normalize=False)
    probe_shape = probe.shape  # (T_probe, C, H, W)
    vid_shape = [n_frames, probe_shape[1], probe_shape[2], probe_shape[3]]

    model = get_best_model(f"models/ref_models/", vid_shape, vid_name, config, device)
    model.eval()

    num_frames = int(n_frames)
    num_batches = (num_frames + batch_size - 1) // batch_size

    # Warmup (not timed)
    with torch.no_grad():
        for _ in range(warmup_iters):
            for batch_idx in range(num_batches):
                min_t = batch_idx * batch_size
                max_t = min((batch_idx + 1) * batch_size, num_frames)
                t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.float32)
                norm_t_batch = t_batch / (num_frames - 1)
                _ = model(norm_t_batch)

    if profile:
        os.makedirs(profile_dir, exist_ok=True)
        trace_dir = os.path.join(profile_dir, profile_name)
        with torch.no_grad(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        ) as prof:
            for batch_idx in range(num_batches):
                min_t = batch_idx * batch_size
                max_t = min((batch_idx + 1) * batch_size, num_frames)
                t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.float32)
                norm_t_batch = t_batch / (num_frames - 1)
                _ = model(norm_t_batch)
                prof.step()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    total_frames = 0

    with torch.no_grad():
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            starter.record()
            for batch_idx in range(num_batches):
                min_t = batch_idx * batch_size
                max_t = min((batch_idx + 1) * batch_size, num_frames)
                t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.float32)
                norm_t_batch = t_batch / (num_frames - 1)
                _ = model(norm_t_batch)
            ender.record()
            torch.cuda.synchronize(device)
            total_ms += starter.elapsed_time(ender)
            total_frames += num_frames

    avg_ms = total_ms / repeats
    fps = (total_frames / repeats) / (avg_ms / 1000.0)
    print(f"Benchmark FPS (CUDA events): {fps:.2f} fps | avg time: {avg_ms:.2f} ms per {num_frames} frames")
    if profile:
        print(f"Profiler trace saved to: {trace_dir}")


def make_mp4(png_frame_dir, output_path="output.mp4", base_name="pred_frame", fps=24):
    # Assumes frames are named in order: frame000.png, frame001.png, ...
    input_pattern = os.path.join(png_frame_dir, f"{base_name}%03d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, check=True)


def module_visualization(basedir, vid_name, n_frames, config, device, variants=None, batch_size=6, commit="3db530e"):
    if variants is None:
        variants = ['baseline', 'only_real_grid', 'only_realt', 'only_complex_grid', 'only_complext', 'temporal_operators']

    # Determine model input shape using a single-frame probe (avoids loading full video)
    probe = load_video_frames(f"{basedir}/{vid_name}", device, max_frames=1, dtype=torch.uint8, normalize=False)
    probe_shape = probe.shape  # (T_probe, C, H, W)
    # Use the provided `n_frames` for temporal length, but match spatial/channel dims from probe
    vid_shape = [n_frames, probe_shape[1], probe_shape[2], probe_shape[3]]
    model = get_best_model(f"models/ref_models/{commit}/{config}", vid_shape, vid_name, config, device)
    model.eval()

    num_frames = int(n_frames)
    num_batches = (num_frames + batch_size - 1) // batch_size

    for v in variants:
        # create main preds dir for variant
        os.makedirs(f"visuals/{vid_name}/{config}/{v}/preds", exist_ok=True)
        # if storing forward/back separately, create subfolders
        if v == 'temporal_operators':
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/minus_one/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/plus_one/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/minus_two/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/plus_two/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/full_operator_residual/preds", exist_ok=True)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            print(f"Processing batch {batch_idx + 1}/{num_batches} for variants: {variants}")
            min_t = batch_idx * batch_size
            max_t = min((batch_idx + 1) * batch_size, num_frames)
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            norm_t_batch = t_batch.float() / (num_frames - 1)  # normalize time for operators that use it

            # baseline call
            if 'baseline' in variants:
                out_base = model(norm_t_batch)

            # zeroed variants use forward flags
            if 'only_real_grid' in variants:
                out_real_grid = model(norm_t_batch, zero_real_tucker=True, zero_complex_tucker=True, zero_complex_grid=True)
            if 'only_realt' in variants:
                out_realt = model(norm_t_batch, zero_feature_grid=True, zero_complex_tucker=True, zero_complex_grid=True)
            if 'only_complex_grid' in variants:
                out_complex_grid = model(norm_t_batch, zero_real_tucker=True, zero_feature_grid=True, zero_complex_tucker=True)
            if 'only_complext' in variants:
                out_complext = model(norm_t_batch, zero_real_tucker=True, zero_feature_grid=True, zero_complex_grid=True)

            # forward/backward operators passed through upres
            if 'temporal_operators' in variants:
                # model.forward(..., return_operators=True) -> (refined, refined_forward, refined_backward)
                _, minus_one, plus_one, minus_two, plus_two = model(norm_t_batch, return_operators=True)
                full_operator_residual = minus_one + plus_one + minus_two + plus_two

            # save per-variant frames
            for i in range(t_batch.shape[0]):
                idx = min_t + i
                if 'baseline' in variants:
                    save_image(out_base[i], f"visuals/{vid_name}/{config}/baseline/preds/pred_frame{idx:03d}.png")
                if 'only_real_grid' in variants:
                    save_image(out_real_grid[i], f"visuals/{vid_name}/{config}/only_real_grid/preds/pred_frame{idx:03d}.png")
                if 'only_realt' in variants:
                    save_image(out_realt[i], f"visuals/{vid_name}/{config}/only_realt/preds/pred_frame{idx:03d}.png")
                if 'only_complex_grid' in variants:
                    save_image(out_complex_grid[i], f"visuals/{vid_name}/{config}/only_complex_grid/preds/pred_frame{idx:03d}.png")
                if 'only_complext' in variants:
                    save_image(out_complext[i], f"visuals/{vid_name}/{config}/only_complext/preds/pred_frame{idx:03d}.png")
                if 'temporal_operators' in variants:
                    save_image(minus_one[i], f"visuals/{vid_name}/{config}/temporal_operators/minus_one/preds/pred_frame{idx:03d}.png")
                    save_image(plus_one[i], f"visuals/{vid_name}/{config}/temporal_operators/plus_one/preds/pred_frame{idx:03d}.png")
                    save_image(minus_two[i], f"visuals/{vid_name}/{config}/temporal_operators/minus_two/preds/pred_frame{idx:03d}.png")
                    save_image(plus_two[i], f"visuals/{vid_name}/{config}/temporal_operators/plus_two/preds/pred_frame{idx:03d}.png")
                    save_image(full_operator_residual[i], f"visuals/{vid_name}/{config}/temporal_operators/full_operator_residual/preds/pred_frame{idx:03d}.png")

    # make mp4s for each variant
    for v in variants:
        try:
            if v == 'temporal_operators':
                for op in ['minus_one', 'plus_one', 'minus_two', 'plus_two', 'full_operator_residual']:
                    src_dir = f"visuals/{vid_name}/{config}/{v}/{op}/preds"
                    out_path = f"visuals/{vid_name}/{config}/{v}/{op}.mp4"
                    make_mp4(src_dir, output_path=out_path, base_name="pred_frame", fps=24)

            else:
                src_dir = f"visuals/{vid_name}/{config}/{v}/preds"
                out_path = f"visuals/{vid_name}/{config}/{v}.mp4"
                make_mp4(src_dir, output_path=out_path, base_name="pred_frame", fps=24)
        except Exception as e:
            print(f"Failed to create mp4 for {v}: {e}")


if __name__ == "__main__":
    device = "cuda:1"
    name = "bunny"
    config = "small"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # module_visualization("static/benchmarks/uvg", name, n_frames=300, config=config, device=device)
    benchmark_fps_torch_events("static/benchmarks", name, config, device, n_frames=132, batch_size=1)
    # benchmark_psnr("static/benchmarks/uvg", name, config, device)
    # make_mp4(f"visuals/{name}/{config}/preds", output_path=f"visuals/{name}/{config}/preds/output.mp4", base_name="pred_frame", fps=24)
    # make_mp4(f"visuals/{name}/{config}/residual", output_path=f"visuals/{name}/{config}/residual/output.mp4", base_name="residual_frame", fps=24)
