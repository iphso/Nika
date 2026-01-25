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
    state_dict = torch.load(all_models[-1], map_location=device)
    model.load_state_dict(state_dict)
    return model


def benchmark_psnr(basedir, vid_name, config, device):
    vid = load_video_frames(f"{basedir}/{vid_name}", device, max_frames=600, dtype=torch.uint8, normalize=False)
    model = get_best_model("models", vid.shape, vid_name, config, device)
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
                if (residual_max - residual_min) > 1e-8:
                    res_norm = (res - residual_min) / (residual_max - residual_min)
                else:
                    res_norm = torch.zeros_like(res)
                save_image(res_norm, f"visuals/{vid_name}/{config}/residual/residual_frame{min_t + i:03d}.png")
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
    device = "cuda:0"
    name = "bunny"
    config = "xxs"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    benchmark_psnr("static/benchmarks", name, config, device)
    make_mp4(f"visuals/{name}/{config}/preds", output_path=f"visuals/{name}/{config}/preds/output.mp4", base_name="pred_frame", fps=24)
    make_mp4(f"visuals/{name}/{config}/residual", output_path=f"visuals/{name}/{config}/residual/output.mp4", base_name="residual_frame", fps=24)
