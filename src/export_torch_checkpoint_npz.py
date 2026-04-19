from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    state = _clean_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    arrays = {key: value.detach().cpu().numpy() for key, value in state.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    print(output)
    print(len(arrays))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
