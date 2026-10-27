"""
Wall-clock benchmark hook, dropped verbatim into every method's repo.

The budget is measured in training time: the clock starts at the first optimizer step and
is paused while a snapshot is written, so process startup (imports, data loading, model
build) and snapshot I/O count for no one, and no method trains in someone else's eval.
Evaluation never runs inside training; every snapshot is evaluated afterwards (NB_EVAL_DIR).

The training loop calls poll() at the top of every optimizer step. When the training clock
has passed the next mark (NB_MARKS, minutes), poll() saves the weights as they stand, so a
snapshot is the state at the first step boundary after its mark, for every method alike.
After the last mark the process exits. A trainer whose schedule ends before the budget
calls final(), whose weights then stand for every later mark.

NB_MEM_GIB caps GPU memory for every method alike (the envelope of a smaller card).

Inactive unless NB_SNAP_DIR is set, so the repos behave natively otherwise.
"""
import json
import os
import sys
import time

import torch

SNAP_DIR = os.environ.get("NB_SNAP_DIR")
ACTIVE = bool(SNAP_DIR)
T0 = float(os.environ.get("NB_T0", time.time()))
MARKS = [float(m) * 60 for m in os.environ.get("NB_MARKS", "").split(",") if m]

_n_saved = 0
_started = None     # wall time of the first optimizer step
_paused = 0.0       # seconds spent writing snapshots, excluded from the training clock


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_elapsed():
    """Seconds of training so far (0 before the first step)."""
    return 0.0 if _started is None else time.time() - _started - _paused


def install():
    """Call once at startup. Also fixes the precision policy for every method."""
    # Matched numerics: Nika already runs TF32 matmuls/convs; give every method the same.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Matched memory envelope: NB_MEM_GIB caps the CUDA caching allocator, so a method that needs
    # more than the envelope OOMs here exactly as it would on the smaller card.
    cap = os.environ.get("NB_MEM_GIB")
    if cap and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(float(cap) * 2**30 / total, 0)
        print(f"NB_MEM cap_gib={float(cap)} of {total / 2**30:.1f}", flush=True)
    if ACTIVE:
        os.makedirs(SNAP_DIR, exist_ok=True)
        print(f"NB_HOOK active snap_dir={SNAP_DIR} marks_s={MARKS}", flush=True)


def _save(state_fn, tag, epoch, mark_s):
    """Write a snapshot with the training clock stopped."""
    global _paused
    _sync()                       # queued kernels belong to training: finish them on the clock
    t_train = train_elapsed()
    t = time.time()
    state = {k: v.detach().cpu() for k, v in state_fn().items()}
    path = os.path.join(SNAP_DIR, f"{tag}.pt")
    torch.save({"state_dict": state, "epoch": epoch, "t_train": t_train, "mark_s": mark_s}, path + ".tmp")
    os.replace(path + ".tmp", path)
    _paused += time.time() - t
    print(f"NB_SNAP {tag} epoch={epoch} t_train={t_train:.2f} mark_s={mark_s} "
          f"t_wall={time.time() - T0:.2f} save_s={time.time() - t:.2f}", flush=True)


def _done():
    print(f"NB_DONE t_train={train_elapsed():.2f} paused_s={_paused:.2f}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def poll(state_fn, epoch):
    """Top of every optimizer step, before the forward pass."""
    global _n_saved, _started
    if not ACTIVE:
        return
    if _started is None:
        _sync()
        _started = time.time()
        print(f"NB_TRAIN_START t={_started - T0:.2f}", flush=True)
    while _n_saved < len(MARKS) and train_elapsed() >= MARKS[_n_saved]:
        _save(state_fn, f"snap_{_n_saved:02d}", epoch, MARKS[_n_saved])
        _n_saved += 1
    if MARKS and _n_saved == len(MARKS):
        _done()


def epoch_end(epoch, frames):
    """After each full pass over the video (epoch = number of completed epochs)."""
    if ACTIVE:
        _sync()
        print(f"NB_EPOCH {epoch} frames={frames} t={train_elapsed():.3f}", flush=True)


def final(state_fn, epoch):
    """Training finished before the budget: these weights stand for every remaining mark."""
    if not ACTIVE:
        return
    _save(state_fn, "snap_final", epoch, None)
    _done()


# ----------------------------------------------------------------------------------------
# Eval mode: NB_EVAL_DIR=<run dir>. The trainer builds its model and data exactly as for
# training, then calls evaluate() instead of training. One metric, one ground-truth loader
# and one timing protocol for every method.
# ----------------------------------------------------------------------------------------
EVAL_DIR = os.environ.get("NB_EVAL_DIR")


def load_gt(frame_dir):
    """uint8 [T,3,H,W] from the sorted PNGs, identical for every method."""
    import glob
    import numpy as np
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    return torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])).permute(0, 3, 1, 2).contiguous()


def frame_psnr(pred, gt):
    """Per-frame PSNR of predictions clamped to [0,1] against float GT in [0,1], both [B,3,H,W]."""
    mse = (pred.float().clamp(0, 1) - gt.float()).pow(2).flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


@torch.no_grad()
def _decode_fps(fn, n_frames, batch, device, warmup_passes=2, passes=5):
    idx_batches = [torch.arange(s, min(s + batch, n_frames), device=device) for s in range(0, n_frames, batch)]
    for _ in range(warmup_passes):
        for idx in idx_batches:
            fn(idx)
    torch.cuda.synchronize(device)
    t = time.perf_counter()
    for _ in range(passes):
        for idx in idx_batches:
            fn(idx)
    torch.cuda.synchronize(device)
    return passes * n_frames / (time.perf_counter() - t)


@torch.no_grad()
def evaluate(model, decode, frame_dir, device, load=None, decode_only=None, n_params=None, psnr_batch=1):
    """
    decode(idx) -> [B,3,H,W] float prediction for frame indices idx (LongTensor on device).
    decode_only: optional factory returning the decode-time function used for FPS (e.g. HNeRV
        decodes from stored embeddings, so its encoder is excluded); defaults to decode.
    n_params: the method's own parameter-count convention, if it differs from state_dict numel.
    """
    load = load or (lambda sd: model.load_state_dict(sd))
    run = json.load(open(os.path.join(EVAL_DIR, "run.json")))
    gt = load_gt(frame_dir)
    T = gt.shape[0]
    model.eval()
    out = dict(n_frames=T, hw=list(gt.shape[2:]), n_params=n_params,
               state_numel=sum(v.numel() for v in model.state_dict().values() if v.is_floating_point()),
               marks=[])
    cache = {}
    for m in run["marks"]:
        rec = dict(m)
        if m.get("snap"):
            if m["snap"] not in cache:
                ck = torch.load(os.path.join(EVAL_DIR, "snaps", m["snap"] + ".pt"), map_location=device)
                load(ck["state_dict"])
                model.eval()
                psnrs = []
                for s in range(0, T, psnr_batch):
                    idx = torch.arange(s, min(s + psnr_batch, T), device=device)
                    pred = decode(idx)
                    psnrs.append(frame_psnr(pred, gt[s:s + len(idx)].to(device) / 255.0).cpu())
                psnrs = torch.cat(psnrs)
                cache[m["snap"]] = dict(psnr=psnrs.mean().item(), psnr_min=psnrs.min().item())
            rec.update(cache[m["snap"]])
        out["marks"].append(rec)
        print(f"NB_EVAL mark={m['mark_min']} snap={m.get('snap')} psnr={rec.get('psnr')}", flush=True)

    # Decode speed with the last snapshot's weights (speed does not depend on the values).
    fps_fn = decode_only() if decode_only else decode
    out["decode_fps"] = {}
    for b in (1, 8):
        try:
            out["decode_fps"][f"bs{b}"] = _decode_fps(fps_fn, T, b, device)
        except torch.OutOfMemoryError:
            out["decode_fps"][f"bs{b}"] = None
            torch.cuda.empty_cache()
    print(f"NB_EVAL decode_fps={out['decode_fps']}", flush=True)
    with open(os.path.join(EVAL_DIR, "eval.json"), "w") as f:
        json.dump(out, f, indent=1)


COUNT_ONLY = bool(os.environ.get("NB_COUNT_ONLY"))


def count_exit(n_params):
    """NB_COUNT_ONLY=1: report the method's own parameter count right after model build, then stop."""
    if COUNT_ONLY:
        print(f"NB_PARAMS {int(n_params)}", flush=True)
        raise SystemExit(0)
