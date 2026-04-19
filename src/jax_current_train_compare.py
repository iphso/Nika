from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_current_nika import batch_psnr_loss, infer_spec, load_params
from jax_soap import SOAPHyperParams, debug_parameter_step, debug_post_state, init_state, step as soap_step


THRESHOLDS = {
    "loss": {"max_abs": 2e-4},
    "prediction": {"max_abs": 2e-3, "mean_rel_nontrivial": 5e-4},
    "gradients": {"max_abs": 6e-3, "mean_rel_nontrivial": 2e-3},
    "params": {"max_abs": 6e-3, "mean_rel_nontrivial": 2e-3},
}


def _stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    abs_diff = np.abs(actual - reference)
    rel = abs_diff / np.maximum(np.abs(reference), 1e-6)
    significant = np.abs(reference) > 1e-3
    if significant.any():
        rel_nontrivial = abs_diff[significant] / np.abs(reference[significant])
        mean_rel_nontrivial = float(rel_nontrivial.mean())
        max_rel_nontrivial = float(rel_nontrivial.max())
    else:
        mean_rel_nontrivial = 0.0
        max_rel_nontrivial = 0.0
    return {
        "mean_abs": float(abs_diff.mean()),
        "max_abs": float(abs_diff.max()),
        "mean_rel": float(rel.mean()),
        "max_rel": float(rel.max()),
        "mean_rel_nontrivial": mean_rel_nontrivial,
        "max_rel_nontrivial": max_rel_nontrivial,
    }


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _to_numpy_tree(tree: dict[str, jnp.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in sorted(tree.items())}


def _named_stats(actual: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, object]:
    total_abs = 0.0
    total_count = 0
    worst_max_abs = -1.0
    worst_rel = -1.0
    worst_abs_name = ""
    worst_rel_name = ""

    for name in sorted(reference):
        stats = _stats(actual[name].astype(reference[name].dtype, copy=False), reference[name])
        count = int(reference[name].size)
        total_abs += stats["mean_abs"] * count
        total_count += count
        if stats["max_abs"] > worst_max_abs:
            worst_max_abs = stats["max_abs"]
            worst_abs_name = name
        if stats["mean_rel_nontrivial"] > worst_rel:
            worst_rel = stats["mean_rel_nontrivial"]
            worst_rel_name = name

    return {
        "mean_abs": float(total_abs / max(total_count, 1)),
        "max_abs": float(worst_max_abs),
        "mean_rel_nontrivial": float(worst_rel),
        "worst_max_abs_name": worst_abs_name,
        "worst_mean_rel_nontrivial_name": worst_rel_name,
    }


def _check(kind: str, stats: dict[str, object]) -> bool:
    threshold = THRESHOLDS[kind]
    max_abs_ok = float(stats["max_abs"]) <= threshold["max_abs"]
    mean_rel_ok = True
    if "mean_rel_nontrivial" in threshold:
        mean_rel_ok = float(stats["mean_rel_nontrivial"]) <= threshold["mean_rel_nontrivial"]
    return max_abs_ok and mean_rel_ok


def _param_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "__")


def _compare_debug_arrays(step_idx: int, param_name: str, actual: dict[str, jnp.ndarray], reference_path: Path) -> None:
    with np.load(reference_path) as data:
        reference = {key: data[key] for key in data.files}
    for key in sorted(reference):
        if key not in actual:
            print(json.dumps({"step": step_idx, "component": "soap_debug", "param": param_name, "tensor": key, "missing": True}))
            continue
        stats = _stats(np.asarray(actual[key]).astype(reference[key].dtype, copy=False), reference[key])
        print(json.dumps({"step": step_idx, "component": "soap_debug", "param": param_name, "tensor": key, **stats}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir)
    metadata = json.loads((reference_dir / "metadata.json").read_text())

    params = load_params(reference_dir / "initial_params.npz")
    spec = infer_spec(params)
    targets = jnp.asarray(np.load(reference_dir / "targets.npy"), dtype=jnp.float32)
    norm_t = jnp.asarray(np.load(reference_dir / "norm_t.npy"), dtype=jnp.float32)

    optimizer = metadata["optimizer"]
    hparams = SOAPHyperParams(
        lr=float(optimizer["lr"]),
        betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
        eps=float(optimizer["eps"]),
        weight_decay=float(optimizer["weight_decay"]),
        precondition_frequency=int(optimizer["precondition_frequency"]),
        max_precond_dim=int(optimizer["max_precond_dim"]),
        merge_dims=bool(optimizer["merge_dims"]),
        precondition_1d=bool(optimizer["precondition_1d"]),
        normalize_grads=bool(optimizer["normalize_grads"]),
        data_format=str(optimizer["data_format"]),
        correct_bias=bool(optimizer["correct_bias"]),
    )
    opt_state = init_state(params, hparams)
    watch_params = metadata.get("watch_params", [])
    watch_steps = set(metadata.get("watch_steps", []))

    overall_ok = True
    loss_and_grad = jax.value_and_grad(batch_psnr_loss, has_aux=True)

    for step_idx in range(targets.shape[0]):
        step_dir = reference_dir / f"step_{step_idx:03d}"
        debug_pre: dict[str, dict[str, jnp.ndarray]] = {}
        ((loss, aux), grads) = loss_and_grad(
            params,
            norm_t[step_idx:step_idx + 1],
            targets[step_idx:step_idx + 1],
            spec,
        )

        metrics = json.loads((step_dir / "metrics.json").read_text())
        loss_stats = {"max_abs": float(abs(float(loss) - float(metrics["loss"])))}
        loss_ok = _check("loss", loss_stats)
        overall_ok = overall_ok and loss_ok
        print(json.dumps({"step": step_idx, "component": "loss", "ok": loss_ok, **loss_stats}))

        reference_prediction = np.load(step_dir / "prediction.npy")
        prediction_stats = _stats(np.asarray(aux["predictions"]).astype(reference_prediction.dtype, copy=False), reference_prediction)
        prediction_ok = _check("prediction", prediction_stats)
        overall_ok = overall_ok and prediction_ok
        print(json.dumps({"step": step_idx, "component": "prediction", "ok": prediction_ok, **prediction_stats}))

        reference_gradients = _load_npz_dict(step_dir / "gradients.npz")
        gradient_stats = _named_stats(_to_numpy_tree(grads), reference_gradients)
        gradients_ok = _check("gradients", gradient_stats)
        overall_ok = overall_ok and gradients_ok
        print(json.dumps({"step": step_idx, "component": "gradients", "ok": gradients_ok, **gradient_stats}))

        if step_idx in watch_steps:
            for name in watch_params:
                if name not in params or name not in grads or name not in opt_state:
                    continue
                debug_arrays, debug_meta = debug_parameter_step(params[name], grads[name], opt_state[name], hparams)
                debug_pre[name] = debug_arrays
                print(json.dumps({"step": step_idx, "component": "soap_debug_meta_pre", "param": name, **debug_meta}))

        params, opt_state = soap_step(params, grads, opt_state, hparams)
        reference_params = _load_npz_dict(step_dir / "post_params.npz")
        param_stats = _named_stats(_to_numpy_tree(params), reference_params)
        params_ok = _check("params", param_stats)
        overall_ok = overall_ok and params_ok
        print(json.dumps({"step": step_idx, "component": "params", "ok": params_ok, **param_stats}))

        if step_idx in watch_steps:
            debug_root = step_dir / "soap_debug"
            for name in watch_params:
                reference_path = debug_root / f"{_param_slug(name)}.npz"
                if not reference_path.exists() or name not in params or name not in opt_state:
                    continue
                debug_post_arrays, debug_post_meta = debug_post_state(params[name], opt_state[name])
                debug_arrays = dict(debug_pre.get(name, {}))
                debug_arrays.update(debug_post_arrays)
                print(json.dumps({"step": step_idx, "component": "soap_debug_meta_post", "param": name, **debug_post_meta}))
                _compare_debug_arrays(step_idx, name, debug_arrays, reference_path)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
