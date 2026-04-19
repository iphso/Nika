from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jax_static_nika import forward_with_intermediates, infer_spec, load_reference_params


DEFAULT_THRESHOLDS = {
    "grid_features": {"max_abs": 1e-4, "mean_rel_nontrivial": 1e-4},
    "real_tucker": {"max_abs": 1e-4, "mean_rel_nontrivial": 1e-4},
    "complex_tucker": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "base_input": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "groupnorm": {"max_abs": 2e-4, "mean_rel_nontrivial": 2e-4},
    "output": {"max_abs": 1e-3, "mean_rel_nontrivial": 2e-4},
}


def _diff_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    abs_diff = np.abs(actual - reference)
    rel_diff = abs_diff / np.maximum(np.abs(reference), 1e-6)
    significant = np.abs(reference) > 1e-3
    if significant.any():
        rel_nontrivial = abs_diff[significant] / np.abs(reference[significant])
        max_rel_nontrivial = float(rel_nontrivial.max())
        mean_rel_nontrivial = float(rel_nontrivial.mean())
    else:
        max_rel_nontrivial = 0.0
        mean_rel_nontrivial = 0.0
    return {
        "mean_abs": float(abs_diff.mean()),
        "max_abs": float(abs_diff.max()),
        "mean_rel": float(rel_diff.mean()),
        "max_rel": float(rel_diff.max()),
        "mean_rel_nontrivial": mean_rel_nontrivial,
        "max_rel_nontrivial": max_rel_nontrivial,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        default="tmp/reference_params/small_bunny_static_model.npz",
    )
    parser.add_argument(
        "--activations",
        default="tmp/reference_activations",
    )
    args = parser.parse_args()

    params = load_reference_params(args.params)
    spec = infer_spec(params)

    activation_dir = Path(args.activations)
    meta = json.loads((activation_dir / "metadata.json").read_text())
    frame_idx = np.asarray([meta["probe_frame"]], dtype=np.int32)

    outputs = {
        key: np.asarray(value)
        for key, value in forward_with_intermediates(params, frame_idx, spec).items()
    }

    overall_ok = True
    for name, threshold in DEFAULT_THRESHOLDS.items():
        reference = np.load(activation_dir / f"{name}.npy")
        actual = outputs[name].astype(np.float32)
        stats = _diff_stats(actual, reference)
        ok = (
            stats["max_abs"] <= threshold["max_abs"]
            and stats["mean_rel_nontrivial"] <= threshold["mean_rel_nontrivial"]
        )
        overall_ok = overall_ok and ok
        print(
            json.dumps(
                {
                    "component": name,
                    "ok": ok,
                    "threshold": threshold,
                    **stats,
                }
            )
        )

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
