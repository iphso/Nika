from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jax_current_nika import forward_with_intermediates, infer_spec, load_params, prepare_runtime


THRESHOLDS = {
    "real_tucker": {"max_abs": 1e-4, "mean_rel_nontrivial": 1e-4},
    "grid_features": {"max_abs": 1e-4, "mean_rel_nontrivial": 1e-4},
    "complex_tucker_construct": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "complex_tucker_grid_real": {"max_abs": 1e-4, "mean_rel_nontrivial": 1e-4},
    "complex_tucker_grid_imag": {"max_abs": 1e-4, "mean_rel_nontrivial": 1e-4},
    "complex_tucker": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "response_input": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "groupnorm": {"max_abs": 4e-4, "mean_rel_nontrivial": 2e-4},
    "operator_input": {"max_abs": 4e-4, "mean_rel_nontrivial": 2e-4},
    "operator_initial": {"max_abs": 3e-4, "mean_rel_nontrivial": 2e-4},
    "operator_time_emb": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "operator_gamma": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "operator_beta": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "operator_output": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "aggregated": {"max_abs": 4e-4, "mean_rel_nontrivial": 2e-4},
    "output": {"max_abs": 1e-3, "mean_rel_nontrivial": 2e-4},
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--activations", required=True)
    args = parser.parse_args()

    params = load_params(args.params)
    spec = infer_spec(params)
    runtime = prepare_runtime(params, spec)
    activation_dir = Path(args.activations)
    meta = json.loads((activation_dir / "metadata.json").read_text())
    norm_t = np.asarray([meta["frame_idx"] / (meta["n_frames"] - 1)], dtype=np.float32)

    raw_outputs = forward_with_intermediates(runtime, norm_t, spec)
    outputs = {key: np.asarray(value) for key, value in raw_outputs.items() if key != "complex_tucker_grid"}
    outputs["complex_tucker_grid_real"] = np.asarray(raw_outputs["complex_tucker_grid"].real)
    outputs["complex_tucker_grid_imag"] = np.asarray(raw_outputs["complex_tucker_grid"].imag)

    overall_ok = True
    for name, threshold in THRESHOLDS.items():
        reference = np.load(activation_dir / f"{name}.npy")
        actual = outputs[name].astype(reference.dtype, copy=False)
        stats = _stats(actual, reference)
        ok = stats["max_abs"] <= threshold["max_abs"] and stats["mean_rel_nontrivial"] <= threshold["mean_rel_nontrivial"]
        overall_ok = overall_ok and ok
        print(json.dumps({"component": name, "ok": ok, "threshold": threshold, **stats}))

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
