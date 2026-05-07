import glob
import math
import copy
import os
import shutil
import re
import time

import torch
import torch.nn.functional as F
import argparse
from torchvision.utils import save_image
import imageio.v3 as iio

from load_data import load_video_frames
from nika import MosaicNika
from soap import SOAP
from configs import REFERENCES

try:
    from fvcore.nn.flop_count import FlopCountAnalysis as FlopCounterMode
except Exception:
    FlopCounterMode = None


def get_best_model(model_dir, vid_shape, vid_name, config, device):
    model_vid = os.path.basename(vid_name)
    candidates = glob.glob(f"{model_dir}/{config}-{model_vid}-*.torch")

    def extract_psnr(path):
        match = re.search(r'psnr([0-9]+(?:\.[0-9]+)?)', path)
        if match:
            return float(match.group(1))
        return float('-inf')

    actual_config = config
    if not candidates:
        fallback = glob.glob(f"{model_dir}/*-{model_vid}-*.torch")
        if not fallback:
            raise RuntimeError(f"No models found for {vid_name} with config {config} in {model_dir}")
        fallback.sort(key=extract_psnr)
        model_path = fallback[-1]
        prefix = os.path.basename(model_path).split(f"-{model_vid}-")[0]
        if prefix not in REFERENCES:
            raise RuntimeError(
                f"Found model file {model_path} but inferred prefix '{prefix}' is not in REFERENCES"
            )
        actual_config = prefix
        print(f"Info: using config '{actual_config}' inferred from {model_path}")
    else:
        candidates.sort(key=extract_psnr)
        model_path = candidates[-1]
        print(f"Best model for {vid_name} with config {config}: {model_path}")

    model_kwargs = dict(REFERENCES[actual_config])
    num_segments = model_kwargs.pop('num_segments', 1)
    if os.path.basename(vid_name) == 'bunny' and 'grid_ranks' in model_kwargs:
        model_kwargs['grid_ranks'] = model_kwargs['grid_ranks'] * 2

    model = MosaicNika(
        target_shape=[4, vid_shape[2], vid_shape[3], vid_shape[0]],
        k=4,
        model_kwargs=model_kwargs,
        out_channels=3,
        device=device,
        num_segments=num_segments,
    )

    state_dict = torch.load(model_path, map_location=device)
    ckpt_map = {k.replace('._orig_mod', ''): v for k, v in state_dict.items()}
    mapped_sd = {}
    for mk in model.state_dict().keys():
        norm_mk = mk.replace('._orig_mod', '')
        if norm_mk in ckpt_map:
            mapped_sd[mk] = ckpt_map[norm_mk]

    missing, unexpected = model.load_state_dict(mapped_sd)
    if missing:
        print(f"Warning: missing keys when loading {model_path}: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading {model_path}: {unexpected}")
    return model


def force_head_order(model, order=('grid', 'real', 'complex')):
    for m in getattr(model, "models", []):
        existing = set(getattr(m, "_head_order", []))
        m._head_order = [x for x in order if x in existing]


def compute_macs(model, device):
    if FlopCounterMode is None:
        return None

    sample_model = model
    if hasattr(model, 'models') and len(model.models) > 0:
        sample_model = model.models[0]
    while hasattr(sample_model, '_orig_mod'):
        sample_model = sample_model._orig_mod

    sample_t = torch.tensor([0.0], device=device, dtype=torch.float32)
    try:
        flops = FlopCounterMode(sample_model, (sample_t,))
        return flops.total()
    except Exception as exc:
        print(f"Warning: could not compute MACs: {exc}")
        return None


def warmup_model(model, num_frames, device, batch_size=8, repeats=10):
    model.eval()
    use_cuda = 'cuda' in device
    if use_cuda:
        torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(repeats):
            if getattr(model, 'num_segments', 1) > 1 and hasattr(model, 'segment_ranges'):
                for segment_id, norm_t, _, _ in iter_segment_batches(model, num_frames, device, batch_size):
                    _ = model.forward_segment(segment_id, norm_t)
            else:
                for start_idx in range(0, num_frames, batch_size):
                    end_idx = min(num_frames, start_idx + batch_size)
                    norm_t = torch.arange(start_idx, end_idx, device=device, dtype=torch.float32) / max(num_frames - 1, 1)
                    _ = model(norm_t)
    if use_cuda:
        torch.cuda.synchronize(device)


def iter_segment_batches(model, num_frames, device, batch_size):
    if getattr(model, 'num_segments', 1) > 1 and hasattr(model, 'segment_ranges'):
        for segment_id, (seg_start, seg_end) in enumerate(model.segment_ranges):
            seg_end = min(seg_end, num_frames)
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            for start_idx in range(seg_start, seg_end, batch_size):
                end_idx = min(start_idx + batch_size, seg_end)
                idx = torch.arange(start_idx, end_idx, device=device, dtype=torch.float32)
                if seg_len > 1:
                    local_t = (idx - seg_start) / (seg_len - 1)
                else:
                    local_t = torch.zeros_like(idx)
                yield segment_id, local_t, start_idx, end_idx
        return
    for start_idx in range(0, num_frames, batch_size):
        end_idx = min(num_frames, start_idx + batch_size)
        idx = torch.arange(start_idx, end_idx, device=device, dtype=torch.float32)
        yield None, idx / max(num_frames - 1, 1), start_idx, end_idx


def measure_decode_fps(model, num_frames, device, batch_size=8, repeats=20):
    model.eval()
    use_cuda = 'cuda' in device

    def batch_iter():
        if getattr(model, 'num_segments', 1) > 1 and hasattr(model, 'segment_ranges'):
            return iter_segment_batches(model, num_frames, device, batch_size)
        return ((None, torch.arange(start_idx, min(start_idx + batch_size, num_frames), device=device, dtype=torch.float32) / max(num_frames - 1, 1), start_idx, min(start_idx + batch_size, num_frames))
                for start_idx in range(0, num_frames, batch_size))

    with torch.no_grad():
        for _ in range(10):
            for segment_id, norm_t, _, _ in batch_iter():
                if segment_id is None:
                    _ = model(norm_t)
                else:
                    _ = model.forward_segment(segment_id, norm_t)

    if use_cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            for segment_id, norm_t, _, _ in batch_iter():
                if segment_id is None:
                    _ = model(norm_t)
                else:
                    _ = model.forward_segment(segment_id, norm_t)

    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    total_frames = float(num_frames * repeats)
    return total_frames / elapsed if elapsed > 0 else float('inf')


def measure_psnr(model, vid, device, batch_size=8, debug_psnr=False):
    model.eval()
    num_frames = vid.shape[0]
    total_psnr = 0.0
    total_frames = 0

    with torch.no_grad():
        if getattr(model, 'num_segments', 1) > 1 and hasattr(model, 'segment_ranges'):
            batch_iter = iter_segment_batches(model, num_frames, device, batch_size)
        else:
            batch_iter = ((None, torch.arange(start_idx, min(start_idx + batch_size, num_frames), device=device, dtype=torch.float32) / max(num_frames - 1, 1), start_idx, min(start_idx + batch_size, num_frames))
                          for start_idx in range(0, num_frames, batch_size))

        for segment_id, norm_t, start_idx, end_idx in batch_iter:
            if segment_id is None:
                pred = model(norm_t)
            else:
                pred = model.forward_segment(segment_id, norm_t)
            pred_clamped = pred.clamp(0.0, 1.0)
            gt = vid[start_idx:end_idx].to(torch.float32) / 255.0

            per_frame_mse = F.mse_loss(pred_clamped, gt, reduction='none')
            per_frame_mse = per_frame_mse.view(per_frame_mse.shape[0], -1).mean(dim=1)
            frame_psnr = 10.0 * torch.log10(1.0 / (per_frame_mse + 1e-8))
            if debug_psnr:
                for idx, psnr_value in enumerate(frame_psnr.tolist(), start=start_idx):
                    print(f"frame {idx}: psnr={psnr_value:.6f}")
            total_psnr += frame_psnr.sum().item()
            total_frames += frame_psnr.shape[0]

    avg_psnr = total_psnr / total_frames if total_frames > 0 else 0.0
    return float(avg_psnr)


def make_encode_model_copy(model):
    base_model = model
    for attr in ('_orig_mod', 'original', 'inner'):
        if hasattr(base_model, attr):
            base_model = getattr(base_model, attr)
    try:
        return copy.deepcopy(base_model)
    except Exception:
        return copy.deepcopy(model)


def measure_encode_fps(model, vid, device, steps=5, n_frames=None, batch_size=8, compile_model=False):
    encode_model = make_encode_model_copy(model)
    if compile_model:
        try:
            encode_model = torch.compile(encode_model)
            if hasattr(encode_model, 'forward_segment'):
                try:
                    encode_model.forward_segment = torch.compile(encode_model.forward_segment)
                except Exception as exc:
                    print(f"Warning: torch.compile failed for encode_model.forward_segment: {exc}")
        except Exception as exc:
            print(f"Warning: torch.compile failed for encode model copy: {exc}")

    encode_model.train()
    if n_frames is None:
        n_frames = vid.shape[0]
    n_frames = min(n_frames, vid.shape[0])
    batch_size = min(batch_size, n_frames)
    opt = SOAP(encode_model.parameters(), lr=1e-2)

    use_cuda = 'cuda' in device
    if use_cuda:
        torch.cuda.synchronize(device)

    use_target_device = vid.device == torch.device(device) if isinstance(device, str) else vid.device == device
    encode_batches = []
    if getattr(model, 'num_segments', 1) > 1 and hasattr(model, 'segment_ranges'):
        for segment_id, (seg_start, seg_end) in enumerate(model.segment_ranges):
            seg_end = min(seg_end, n_frames)
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            for start_idx in range(seg_start, seg_end, batch_size):
                end_idx = min(start_idx + batch_size, seg_end)
                idx = torch.arange(start_idx, end_idx, device=device, dtype=torch.float32)
                if seg_len > 1:
                    norm_t = (idx - seg_start) / (seg_len - 1)
                else:
                    norm_t = torch.zeros_like(idx)
                target = vid[start_idx:end_idx].float() / 255.0
                if not use_target_device:
                    target = target.to(device)
                encode_batches.append((segment_id, norm_t, target))
    else:
        for start_idx in range(0, n_frames, batch_size):
            end_idx = min(start_idx + batch_size, n_frames)
            idx = torch.arange(start_idx, end_idx, device=device, dtype=torch.float32)
            norm_t = idx / max(vid.shape[0] - 1, 1)
            target = vid[start_idx:end_idx].float() / 255.0
            if not use_target_device:
                target = target.to(device)
            encode_batches.append((None, norm_t, target))

    warmup_steps = min(2, steps)
    for _ in range(warmup_steps):
        for segment_id, norm_t, target in encode_batches:
            opt.zero_grad(set_to_none=True)
            if segment_id is None:
                pred = encode_model(norm_t)
            else:
                pred = encode_model.forward_segment(segment_id, norm_t)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()

    if use_cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    for _ in range(steps):
        for segment_id, norm_t, target in encode_batches:
            opt.zero_grad(set_to_none=True)
            if segment_id is None:
                pred = encode_model(norm_t)
            else:
                pred = encode_model.forward_segment(segment_id, norm_t)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()

    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    total_frames = float(n_frames * steps)
    return total_frames / elapsed if elapsed > 0 else float('inf')


def make_mp4(src_dir, output_path, base_name="pred_frame", fps=24):
    files = sorted(glob.glob(os.path.join(src_dir, f"{base_name}*.png")))
    if not files:
        raise RuntimeError(f"No frames found in {src_dir}")
    imgs = []
    for f in files:
        try:
            imgs.append(iio.imread(f))
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    if not imgs:
        raise RuntimeError(f"No readable frames in {src_dir}")
    try:
        iio.imwrite(output_path, imgs, fps=fps)
    except Exception as e:
        # fall back to a simple looped write using ffmpeg via imageio plugins if available
        try:
            iio.imwrite(output_path, imgs, plugin="ffmpeg", fps=fps)
        except Exception as e2:
            raise


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def iter_segment_frame_indices(model, frame_indices, device, batch_size):
    # frame_indices: iterable of integer frame indices (global frame numbers)
    frame_indices = list(frame_indices)

    if getattr(model, 'num_segments', 1) > 1 and hasattr(model, 'segment_ranges'):
        for segment_id, (seg_start, seg_end) in enumerate(model.segment_ranges):
            seg_len = seg_end - seg_start
            seg_frames = [i for i in frame_indices if seg_start <= i < seg_end]
            for j in range(0, len(seg_frames), batch_size):
                batch = seg_frames[j:j + batch_size]
                idx = torch.tensor(batch, device=device, dtype=torch.float32)
                local_t = (idx - seg_start) / max(seg_len - 1, 1)
                yield segment_id, local_t, batch
        return

    # single-segment global mapping
    total = getattr(model, 'total_frames', None)
    if total is None:
        raise RuntimeError("Model does not expose total_frames for single-segment normalization")
    for j in range(0, len(frame_indices), batch_size):
        batch = frame_indices[j:j + batch_size]
        idx = torch.tensor(batch, device=device, dtype=torch.float32)
        norm_t = idx / max(total - 1, 1)
        yield None, norm_t, batch


def module_visualization(basedir, vid_name, n_frames, config, device, variants=None, batch_size=1, commit=None):
    """Render module-ablation visualizations for a trained checkpoint.

    Args:
        basedir: Root directory containing benchmark frame folders.
        vid_name: Video identifier to visualize.
        n_frames: Number of frames to render for the visualization run.
        config: Model preset whose checkpoint should be loaded.
        device: Device on which to run inference.
        variants: Optional list of visualization variants to export.
        batch_size: Number of frames to render per inference batch.
        commit: Optional flag to further specify target model
    """
    if variants is None:
        variants = ['baseline', 'only_real_grid', 'only_realt', 'only_complex_grid', 'only_complext', 'temporal_operators']

    # Load full training video to get the canonical train length, then allow
    # rendering a subset (render_num_frames) while constructing the model
    # with the full training length so segment mapping is consistent.
    try:
        full_vid = load_video_frames(f"{basedir}/{vid_name}", device, dtype=torch.uint8, normalize=False)
    except TypeError:
        full_vid = load_video_frames(f"{basedir}/{vid_name}", device, dtype=torch.uint8, normalize=False)

    train_num_frames = int(full_vid.shape[0])
    render_num_frames = int(n_frames) if n_frames is not None else train_num_frames

    vid_shape = [train_num_frames, int(full_vid.shape[1]), int(full_vid.shape[2]), int(full_vid.shape[3])]
    dir_suff = f"{commit}/{config}" if commit is not None else f"{config}"
    model = get_best_model(f"models/ref_models/{dir_suff}", vid_shape, vid_name, config, device)
    force_head_order(model)
    model.eval()

    num_frames = int(render_num_frames)
    num_batches = (num_frames + batch_size - 1) // batch_size

    # prepare output directories (clear stale preds from previous runs)
    for v in variants:
        base_vdir = f"visuals/{vid_name}/{config}/{v}"
        os.makedirs(base_vdir, exist_ok=True)
        if v == 'temporal_operators':
            reset_dir(os.path.join(base_vdir, 'forward', 'preds'))
            reset_dir(os.path.join(base_vdir, 'backward', 'preds'))
            reset_dir(os.path.join(base_vdir, 'full_operator_residual', 'preds'))
        else:
            reset_dir(os.path.join(base_vdir, 'preds'))

    with torch.no_grad():
        batch_iter = iter_segment_batches(model, num_frames, device, batch_size)
        for segment_id, norm_t_batch, start_idx, end_idx in batch_iter:
            print(f"Processing frames {start_idx}:{end_idx} for variants: {variants}")

            # baseline
            out_base = None
            if 'baseline' in variants:
                if segment_id is None:
                    out_base = model(norm_t_batch)
                else:
                    out_base = model.forward_segment(int(segment_id), norm_t_batch)

            # zeroed variants
            out_real_grid = out_realt = out_complex_grid = out_complext = None
            if 'only_real_grid' in variants:
                if segment_id is None:
                    out_real_grid = model(norm_t_batch, zero_real_tucker=True, zero_complex_tucker=True, zero_complex_grid=True)
                else:
                    out_real_grid = model.forward_segment(int(segment_id), norm_t_batch, zero_real_tucker=True, zero_complex_tucker=True, zero_complex_grid=True)
            if 'only_realt' in variants:
                if segment_id is None:
                    out_realt = model(norm_t_batch, zero_feature_grid=True, zero_complex_tucker=True, zero_complex_grid=True)
                else:
                    out_realt = model.forward_segment(int(segment_id), norm_t_batch, zero_feature_grid=True, zero_complex_tucker=True, zero_complex_grid=True)
            if 'only_complex_grid' in variants:
                if segment_id is None:
                    out_complex_grid = model(norm_t_batch, zero_real_tucker=True, zero_feature_grid=True, zero_complex_tucker=True, zero_complex_grid=False)
                else:
                    out_complex_grid = model.forward_segment(int(segment_id), norm_t_batch, zero_real_tucker=True, zero_feature_grid=True, zero_complex_tucker=True, zero_complex_grid=False)
            if 'only_complext' in variants:
                if segment_id is None:
                    out_complext = model(norm_t_batch, zero_real_tucker=True, zero_feature_grid=True, zero_complex_grid=True)
                else:
                    out_complext = model.forward_segment(int(segment_id), norm_t_batch, zero_real_tucker=True, zero_feature_grid=True, zero_complex_grid=True)

            # temporal operators: request per-segment operator outputs
            fwd_op = bwd_op = full_op = None
            if 'temporal_operators' in variants:
                try:
                    seg_idx = 0 if segment_id is None else int(segment_id)
                    res = model.forward_segment(seg_idx, norm_t_batch, return_operators=True)
                    if isinstance(res, tuple) and len(res) == 3:
                        _, fwd_op, bwd_op = res
                        full_op = fwd_op + bwd_op
                    else:
                        print("Warning: model.return_operators did not return (refined, forward, backward)")
                except Exception as e:
                    print(f"Warning: failed to get temporal operators for frames {start_idx}:{end_idx}: {e}")

            # save per-variant frames for this batch
            B = int(end_idx - start_idx)
            for i in range(B):
                idx = int(start_idx + i)
                if out_base is not None:
                    img = out_base[i].clamp(0.0, 1.0).cpu()
                    save_image(img, f"visuals/{vid_name}/{config}/baseline/preds/pred_frame{idx:03d}.png")
                if out_real_grid is not None:
                    img = out_real_grid[i].clamp(0.0, 1.0).cpu()
                    save_image(img, f"visuals/{vid_name}/{config}/only_real_grid/preds/pred_frame{idx:03d}.png")
                if out_realt is not None:
                    img = out_realt[i].clamp(0.0, 1.0).cpu()
                    save_image(img, f"visuals/{vid_name}/{config}/only_realt/preds/pred_frame{idx:03d}.png")
                if out_complex_grid is not None:
                    img = out_complex_grid[i].clamp(0.0, 1.0).cpu()
                    save_image(img, f"visuals/{vid_name}/{config}/only_complex_grid/preds/pred_frame{idx:03d}.png")
                if out_complext is not None:
                    img = out_complext[i].clamp(0.0, 1.0).cpu()
                    save_image(img, f"visuals/{vid_name}/{config}/only_complext/preds/pred_frame{idx:03d}.png")
                if fwd_op is not None:
                    img = fwd_op[i].clamp(0.0, 1.0).cpu()
                    save_image(img, f"visuals/{vid_name}/{config}/temporal_operators/forward/preds/pred_frame{idx:03d}.png")
                if bwd_op is not None:
                    img = bwd_op[i].clamp(0.0, 1.0).cpu()
                    save_image(img, f"visuals/{vid_name}/{config}/temporal_operators/backward/preds/pred_frame{idx:03d}.png")
                if full_op is not None:
                    img = full_op[i].clamp(0.0, 1.0).cpu()
                    save_image(img, f"visuals/{vid_name}/{config}/temporal_operators/full_operator_residual/preds/pred_frame{idx:03d}.png")

    # make mp4s for each variant
    for v in variants:
        try:
            if v == 'temporal_operators':
                for op in ['forward', 'backward', 'full_operator_residual']:
                    src_dir = f"visuals/{vid_name}/{config}/{v}/{op}/preds"
                    out_path = f"visuals/{vid_name}/{config}/{v}/{op}.mp4"
                    make_mp4(src_dir, output_path=out_path, base_name="pred_frame", fps=24)
            else:
                src_dir = f"visuals/{vid_name}/{config}/{v}/preds"
                out_path = f"visuals/{vid_name}/{config}/{v}.mp4"
                make_mp4(src_dir, output_path=out_path, base_name="pred_frame", fps=24)
        except Exception as e:
            print(f"Failed to create mp4 for {v}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Nika decode/encode benchmark")
    parser.add_argument("--basedir", default="static/benchmarks", help="Base video directory")
    parser.add_argument("--name", required=True, help="Video name (e.g., uvg/beauty)")
    parser.add_argument("--config", default="large", help="Config name from configs.REFERENCES")
    parser.add_argument("--device", default="cuda:0", help="Device to run on, e.g. cuda:0 or cpu")
    parser.add_argument("--model_dir", default="models/ref_models/large", help="Directory containing reference model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for decode/PSNR measurement")
    parser.add_argument("--no-compile-model", action="store_true", help="Disable whole-model compilation for inference")
    parser.add_argument("--debug_psnr", action="store_true", help="Print per-frame PSNR values during decode evaluation")
    parser.add_argument("--visualize", action="store_true", help="Run module_visualization and exit")
    parser.add_argument("--viz_frames", type=int, default=None, help="Number of frames to render for visualization (default: use source video length)")
    parser.add_argument("--viz_batch_size", type=int, default=1, help="Batch size for visualization rendering")
    parser.add_argument("--viz_variants", type=str, default=None, help="Comma-separated visualization variants (default: all)")
    parser.add_argument("--viz_basedir", default=None, help="Base dir override for visualization")
    parser.add_argument("--viz_commit", default=None, help="Optional commit subdir for model lookup")
    args = parser.parse_args()

    device = args.device
    if isinstance(device, str):
        if re.fullmatch(r"\d+", device):
            device = f"cuda:{device}"
        elif re.fullmatch(r"cuda\d+", device):
            device = device.replace('cuda', 'cuda:')

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # If visualization requested, run it and exit early
    if args.visualize:
        variants = None
        if args.viz_variants:
            variants = [v.strip() for v in args.viz_variants.split(',') if v.strip()]
        viz_basedir = args.viz_basedir if args.viz_basedir is not None else args.basedir
        module_visualization(viz_basedir, args.name, args.viz_frames, args.config, device,
                             variants=variants, batch_size=args.viz_batch_size, commit=args.viz_commit)
        return

    vid = load_video_frames(f"{args.basedir}/{args.name}", device, dtype=torch.uint8, normalize=False)
    model = get_best_model(args.model_dir, vid.shape, args.name, args.config, device)
    force_head_order(model)

    compile_model = not args.no_compile_model
    if compile_model:
        try:
            model = torch.compile(model)
            print("Compiled whole model successfully.")
            if hasattr(model, 'forward_segment'):
                try:
                    model.forward_segment = torch.compile(model.forward_segment)
                except Exception as exc:
                    print(f"Warning: torch.compile failed for forward_segment: {exc}")
        except Exception as exc:
            print(f"Warning: torch.compile failed for whole model: {exc}")

    macs = compute_macs(model, device)

    warmup_model(model, vid.shape[0], device, batch_size=args.batch_size)
    decode_fps = measure_decode_fps(model, vid.shape[0], device, batch_size=args.batch_size)
    psnr = measure_psnr(model, vid, device, batch_size=args.batch_size, debug_psnr=args.debug_psnr)
    encode_fps = measure_encode_fps(model, vid, device, batch_size=args.batch_size, compile_model=compile_model)

    print(f"Decode FPS: {decode_fps:.2f}")
    print(f"Encode FPS: {encode_fps:.2f}")
    print(f"MACs: {macs:.2f}")
    print(f"Clamped PSNR: {psnr:.4f}")


if __name__ == "__main__":
    main()
