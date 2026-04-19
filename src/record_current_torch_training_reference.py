from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from configs import REFERENCES
from load_data import load_video_frames
from nika import NikaBlock
from soap import SOAP


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any("_orig_mod" in key for key in state):
        return state
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = key
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]
        clean_key = clean_key.replace("._orig_mod.", ".")
        cleaned[clean_key] = value
    return cleaned


def _tensor_dict_to_numpy(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy() for name, tensor in sorted(tensors.items())}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _named_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters()}


def _parse_frame_indices(raw: str | None, n_frames: int, num_steps: int) -> list[int]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    indices = np.linspace(0, n_frames - 1, num=num_steps, dtype=np.int64)
    return [int(item) for item in indices.tolist()]


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _param_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "__")


def _torch_collect_matrix_arrays(prefix: str, mats: list | tuple) -> tuple[dict[str, np.ndarray], list[int]]:
    arrays: dict[str, np.ndarray] = {}
    indices: list[int] = []
    for idx, mat in enumerate(mats):
        if isinstance(mat, list) and len(mat) == 0:
            continue
        arrays[f"{prefix}{idx}"] = mat.detach().cpu().numpy()
        indices.append(idx)
    return arrays, indices


def _torch_debug_snapshot(optimizer: SOAP, param: torch.nn.Parameter, state: dict, group: dict) -> tuple[dict[str, np.ndarray], dict]:
    arrays: dict[str, np.ndarray] = {
        "param_before": param.detach().cpu().numpy(),
        "grad": param.grad.detach().cpu().numpy(),
    }
    meta = {
        "step_before": int(state.get("step", 0)),
        "had_q": bool(state.get("Q") is not None),
        "precondition_frequency": int(group["precondition_frequency"]),
    }

    gg_prev, gg_prev_indices = _torch_collect_matrix_arrays("gg_prev_", state.get("GG", []))
    arrays.update(gg_prev)
    meta["gg_prev_indices"] = gg_prev_indices

    if state.get("Q") is None:
        return arrays, meta

    q_prev, q_prev_indices = _torch_collect_matrix_arrays("q_prev_", state["Q"])
    arrays.update(q_prev)
    meta["q_prev_indices"] = q_prev_indices

    grad_projected = optimizer.project(
        param.grad,
        state,
        merge_dims=group["merge_dims"],
        max_precond_dim=group["max_precond_dim"],
    )
    beta1, beta2 = group["betas"]
    step_num = int(state["step"]) + 1
    exp_avg_prev = state["exp_avg"].detach().clone()
    exp_avg_sq_prev = state["exp_avg_sq"].detach().clone()
    exp_avg_next = exp_avg_prev * beta1 + grad_projected * (1.0 - beta1)
    exp_avg_sq_next = exp_avg_sq_prev * beta2 + grad_projected.square() * (1.0 - beta2)
    denom = exp_avg_sq_next.sqrt() + group["eps"]

    step_size = float(group["lr"])
    if group["correct_bias"]:
        bias_correction1 = 1.0 - beta1 ** step_num
        bias_correction2 = 1.0 - beta2 ** step_num
        step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

    norm_grad = optimizer.project_back(
        exp_avg_next / denom,
        state,
        merge_dims=group["merge_dims"],
        max_precond_dim=group["max_precond_dim"],
    )
    if group["normalize_grads"]:
        norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)

    param_after_formula = param.detach().clone()
    param_after_formula.add_(norm_grad, alpha=-step_size)
    if group["weight_decay"] > 0.0:
        param_after_formula.add_(param_after_formula, alpha=(-group["lr"] * group["weight_decay"]))

    arrays.update({
        "grad_projected": grad_projected.detach().cpu().numpy(),
        "exp_avg_prev": exp_avg_prev.detach().cpu().numpy(),
        "exp_avg_sq_prev": exp_avg_sq_prev.detach().cpu().numpy(),
        "exp_avg_next": exp_avg_next.detach().cpu().numpy(),
        "exp_avg_sq_next": exp_avg_sq_next.detach().cpu().numpy(),
        "denom": denom.detach().cpu().numpy(),
        "norm_grad": norm_grad.detach().cpu().numpy(),
        "param_after_formula": param_after_formula.detach().cpu().numpy(),
    })
    meta["step_num"] = step_num
    meta["step_size"] = step_size
    return arrays, meta


def _torch_post_state_snapshot(param: torch.nn.Parameter, state: dict) -> tuple[dict[str, np.ndarray], dict]:
    arrays: dict[str, np.ndarray] = {
        "param_after_actual": param.detach().cpu().numpy(),
    }
    meta = {
        "step_after": int(state.get("step", 0)),
        "had_q_after": bool(state.get("Q") is not None),
    }
    if "exp_avg" in state:
        arrays["exp_avg_post"] = state["exp_avg"].detach().cpu().numpy()
    if "exp_avg_sq" in state:
        arrays["exp_avg_sq_post"] = state["exp_avg_sq"].detach().cpu().numpy()
    gg_post, gg_post_indices = _torch_collect_matrix_arrays("gg_post_", state.get("GG", []))
    arrays.update(gg_post)
    meta["gg_post_indices"] = gg_post_indices
    if state.get("Q") is not None:
        q_post, q_post_indices = _torch_collect_matrix_arrays("q_post_", state["Q"])
        arrays.update(q_post)
        meta["q_post_indices"] = q_post_indices
    return arrays, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-frames", type=int, default=132)
    parser.add_argument("--frame-indices")
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--betas", default="0.95,0.95")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--precondition-frequency", type=int, default=10)
    parser.add_argument("--max-precond-dim", type=int, default=10000)
    parser.add_argument("--merge-dims", action="store_true")
    parser.add_argument("--precondition-1d", action="store_true")
    parser.add_argument("--normalize-grads", action="store_true")
    parser.add_argument("--correct-bias", action="store_true", default=True)
    parser.add_argument("--no-correct-bias", action="store_false", dest="correct_bias")
    parser.add_argument(
        "--watch-params",
        default="flow_operator.t_modulator.2.weight,complex_tucker.feature_grid.channel_proj.bias,complex_tucker.UW.U_imag,flow_operator.operator_tail.2.weight,complex_tucker.UT.U_real",
    )
    parser.add_argument("--watch-steps", default="0,1,2")
    args = parser.parse_args()

    beta1, beta2 = (float(item.strip()) for item in args.betas.split(","))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    vid = load_video_frames(
        args.frame_dir,
        device=args.device,
        max_frames=args.n_frames,
        dtype=torch.uint8,
        normalize=False,
    )

    model = NikaBlock(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        **REFERENCES[args.config],
        out_channels=3,
        device=args.device,
    )
    state = _clean_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.load_state_dict(state)
    model.train()

    optimizer = SOAP(
        model.parameters(),
        lr=args.lr,
        betas=(beta1, beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
        precondition_frequency=args.precondition_frequency,
        max_precond_dim=args.max_precond_dim,
        merge_dims=args.merge_dims,
        precondition_1d=args.precondition_1d,
        normalize_grads=args.normalize_grads,
        correct_bias=args.correct_bias,
    )

    frame_indices = _parse_frame_indices(args.frame_indices, int(vid.shape[0]), args.num_steps)
    norm_t = np.asarray(frame_indices, dtype=np.float32) / float(vid.shape[0] - 1)
    targets = (vid[frame_indices].to(torch.float32) / 255.0).detach().cpu().numpy()

    _save_npz(out_dir / "initial_params.npz", _tensor_dict_to_numpy(model.state_dict()))
    np.save(out_dir / "frame_indices.npy", np.asarray(frame_indices, dtype=np.int64))
    np.save(out_dir / "norm_t.npy", norm_t)
    np.save(out_dir / "targets.npy", targets)

    params_by_name = _named_parameters(model)
    watch_params = _parse_csv_list(args.watch_params)
    watch_steps = {int(item) for item in _parse_csv_list(args.watch_steps)}
    metadata = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": args.device,
        "n_frames": int(vid.shape[0]),
        "frame_indices": frame_indices,
        "optimizer": {
            "lr": args.lr,
            "betas": [beta1, beta2],
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "precondition_frequency": args.precondition_frequency,
            "max_precond_dim": args.max_precond_dim,
            "merge_dims": args.merge_dims,
            "precondition_1d": args.precondition_1d,
            "normalize_grads": args.normalize_grads,
            "data_format": "channels_first",
            "correct_bias": args.correct_bias,
        },
        "watch_params": watch_params,
        "watch_steps": sorted(watch_steps),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for step_idx, frame_idx in enumerate(frame_indices):
        target = vid[frame_idx].to(torch.float32) / 255.0
        step_dir = out_dir / f"step_{step_idx:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        optimizer.zero_grad(set_to_none=True)
        current_norm_t = torch.tensor([norm_t[step_idx]], device=args.device, dtype=torch.float32)
        prediction = model(current_norm_t)
        mse = F.mse_loss(prediction.squeeze(0), target)
        psnr = -10.0 * torch.log10(mse + 1e-8)
        loss = (-psnr).mean()
        loss.backward()

        debug_root = step_dir / "soap_debug"
        if step_idx in watch_steps:
            debug_root.mkdir(parents=True, exist_ok=True)
            group = optimizer.param_groups[0]
            for name in watch_params:
                if name not in params_by_name:
                    continue
                watched_param = params_by_name[name]
                debug_arrays, debug_meta = _torch_debug_snapshot(optimizer, watched_param, optimizer.state[watched_param], group)
                debug_path = debug_root / f"{_param_slug(name)}.npz"
                _save_npz(debug_path, debug_arrays)
                (debug_root / f"{_param_slug(name)}.json").write_text(json.dumps({"param_name": name, **debug_meta}, indent=2))

        gradients = {
            name: param.grad.detach().cpu().numpy()
            for name, param in sorted(params_by_name.items())
            if param.grad is not None
        }
        _save_npz(step_dir / "gradients.npz", gradients)
        np.save(step_dir / "prediction.npy", prediction.detach().cpu().numpy())

        optimizer.step()
        _save_npz(step_dir / "post_params.npz", _tensor_dict_to_numpy(model.state_dict()))

        if step_idx in watch_steps:
            for name in watch_params:
                if name not in params_by_name:
                    continue
                watched_param = params_by_name[name]
                debug_path = debug_root / f"{_param_slug(name)}.npz"
                if not debug_path.exists():
                    continue
                with np.load(debug_path) as data:
                    existing = {key: data[key] for key in data.files}
                post_arrays, post_meta = _torch_post_state_snapshot(watched_param, optimizer.state[watched_param])
                existing.update(post_arrays)
                _save_npz(debug_path, existing)
                meta_path = debug_root / f"{_param_slug(name)}.json"
                debug_meta = json.loads(meta_path.read_text())
                debug_meta.update(post_meta)
                meta_path.write_text(json.dumps(debug_meta, indent=2))

        metrics = {
            "step": step_idx,
            "frame_idx": frame_idx,
            "norm_t": float(norm_t[step_idx]),
            "loss": float(loss.item()),
            "mse": float(mse.item()),
            "psnr": float(psnr.item()),
        }
        (step_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics))

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
