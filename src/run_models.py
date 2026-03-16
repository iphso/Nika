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
    model.load_state_dict(state_dict)
    return model


def benchmark_psnr(basedir, vid_name, config, device):
    vid = load_video_frames(f"{basedir}/{vid_name}", device, max_frames=600, dtype=torch.uint8, normalize=False)
    model = get_best_model("models/ref_models/d35975e/{config}", vid.shape, vid_name, config, device)

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

    batch_size = 10
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


def ablation_harness(basedir, vid_name, n_frames, config, device, variants=None, batch_size=10):
    """Run several ablation variants using `NikaBlock.forward` flags and save frames.

    Variants (default):
      - 'baseline'
      - 'no_tucker' (zero both real and complex tucker heads)
      - 'zero_real'
      - 'zero_complex'
      - 'forward_backward_upres' (use the forward/back outputs passed through `upres`)
    """
    if variants is None:
        variants = ['baseline', 'only_grid', 'only_realt', 'only_complext', 'gridless', 'forward_backward_upres']

    # Determine model input shape using a single-frame probe (avoids loading full video)
    probe = load_video_frames(f"{basedir}/{vid_name}", device, max_frames=1, dtype=torch.uint8, normalize=False)
    probe_shape = probe.shape  # (T_probe, C, H, W)
    # Use the provided `n_frames` for temporal length, but match spatial/channel dims from probe
    vid_shape = [n_frames, probe_shape[1], probe_shape[2], probe_shape[3]]
    model = get_best_model(f"models/ref_models/d35975e/{config}", vid_shape, vid_name, config, device)
    model.eval()

    num_frames = int(n_frames)
    num_batches = (num_frames + batch_size - 1) // batch_size

    for v in variants:
        # create main preds dir for variant
        os.makedirs(f"visuals/{vid_name}/{config}/{v}/preds", exist_ok=True)
        # if storing forward/back separately, create subfolders
        if v == 'forward_backward_upres':
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/forward/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/backward/preds", exist_ok=True)

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
            if 'only_grid' in variants:
                out_no_tucker = model(norm_t_batch, zero_real_tucker=True, zero_complex_tucker=True)
            if 'only_realt' in variants:
                out_zero_real = model(norm_t_batch, zero_complex_tucker=True, zero_feature_grid=True)
            if 'only_complext' in variants:
                out_zero_complex = model(norm_t_batch, zero_real_tucker=True, zero_feature_grid=True)
            if 'gridless' in variants:
                out_backless = model(norm_t_batch, zero_feature_grid=True)

            # forward/backward operators passed through upres
            if 'forward_backward_upres' in variants:
                # model.forward(..., return_operators=True) -> (refined, refined_forward, refined_backward)
                _, refined_forward, refined_backward = model(norm_t_batch, return_operators=True)
                out_forward = refined_forward
                out_backward = refined_backward

            # save per-variant frames
            for i in range(t_batch.shape[0]):
                idx = min_t + i
                if 'baseline' in variants:
                    save_image(out_base[i], f"visuals/{vid_name}/{config}/baseline/preds/pred_frame{idx:03d}.png")
                if 'only_grid' in variants:
                    save_image(out_no_tucker[i], f"visuals/{vid_name}/{config}/only_grid/preds/pred_frame{idx:03d}.png")
                if 'only_realt' in variants:
                    save_image(out_zero_real[i], f"visuals/{vid_name}/{config}/only_realt/preds/pred_frame{idx:03d}.png")
                if 'only_complext' in variants:
                    save_image(out_zero_complex[i], f"visuals/{vid_name}/{config}/only_complext/preds/pred_frame{idx:03d}.png")
                if 'gridless' in variants:
                    save_image(out_backless[i], f"visuals/{vid_name}/{config}/gridless/preds/pred_frame{idx:03d}.png")
                if 'forward_backward_upres' in variants:
                    save_image(out_forward[i], f"visuals/{vid_name}/{config}/forward_backward_upres/forward/preds/pred_frame{idx:03d}.png")
                    save_image(out_backward[i], f"visuals/{vid_name}/{config}/forward_backward_upres/backward/preds/pred_frame{idx:03d}.png")

    # make mp4s
    # make mp4s
    for v in variants:
        try:
            if v == 'forward_backward_upres':
                fwd_src = f"visuals/{vid_name}/{config}/{v}/forward/preds"
                bwd_src = f"visuals/{vid_name}/{config}/{v}/backward/preds"
                make_mp4(fwd_src, output_path=f"visuals/{vid_name}/{config}/forward.mp4", base_name="pred_frame", fps=24)
                make_mp4(bwd_src, output_path=f"visuals/{vid_name}/{config}/backward.mp4", base_name="pred_frame", fps=24)
            else:
                src_dir = f"visuals/{vid_name}/{config}/{v}/preds"
                out_path = f"visuals/{vid_name}/{config}/{v}.mp4"
                make_mp4(src_dir, output_path=out_path, base_name="pred_frame", fps=24)
        except Exception as e:
            print(f"Failed to create mp4 for {v}: {e}")


def explain(vid, device):
    batch_size = 50

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
    name = "beauty"
    config = "small"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ablation_harness("static/benchmarks/uvg", name, n_frames=600, config=config, device=device)
    # benchmark_psnr("static/benchmarks/uvg", name, config, device)
    # make_mp4(f"visuals/{name}/{config}/preds", output_path=f"visuals/{name}/{config}/preds/output.mp4", base_name="pred_frame", fps=24)
    # make_mp4(f"visuals/{name}/{config}/residual", output_path=f"visuals/{name}/{config}/residual/output.mp4", base_name="residual_frame", fps=24)
