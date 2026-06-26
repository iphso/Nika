"""
Export a MosaicNika checkpoint to a Kuma .iph package.

Config, H, W, and T are all derived from the checkpoint, not passed in:
  - config is parsed off the checkpoint filename (checkpoints are always saved
    as f"{config}-{dataset}-epoch{N}-psnr{P}.torch", see nika.py's feature_test)
  - H, W are recovered from the spatial parameter shapes (grid_features.grid
    or real_tucker.UH/UW), scaled back up by the fixed k=4 upsample factor
  - T is recovered from the UT tucker factor, minus the operator_steps padding
    NikaBlock adds internally, summed across every segment in the checkpoint

Single-segment configs trace MosaicNika.forward() directly (one static graph,
batch size fixed by --batch-size). Multi-segment configs instead export each
segment independently (via forward_segment) and assemble them with a JS router
snippet through kuma.branching.compile_branching -- kuma-bart's scheduler picks
exactly one segment to run per call, never the others (see kuma-bart's plan notes).

--fps embeds optional playback metadata (fps + derived duration_seconds =
total_frames / fps) in the .iph, so a runtime plays this checkpoint's norm_t: 0->1
sweep back at the speed it was actually authored for -- not recoverable from the
checkpoint itself, defaults to 24 (matching run_models.py's make_mp4 default).

Usage:
  python3 kuma_export.py --model models/test_models/xxs4-bunny-epoch1974-psnr39.62.torch --out artifacts/nika.iph
  python3 kuma_export.py --model models/test_models/xxs4-bunny-epoch1974-psnr39.62.torch --out artifacts/nika.iph --fps 30
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from nika import MosaicNika
from configs import REFERENCES


_K = 4  # fixed upsampling factor used throughout this codebase


def _infer_config(checkpoint_path: Path) -> str:
    stem = checkpoint_path.stem
    for name in sorted(REFERENCES, key=len, reverse=True):
        if stem == name or stem.startswith(name + "-"):
            return name
    raise ValueError(
        f"Can't infer a config from checkpoint filename '{checkpoint_path.name}'. "
        f"Expected it to start with one of: {sorted(REFERENCES)}"
    )


def _infer_dataset(checkpoint_path: Path, config: str) -> str:
    rest = checkpoint_path.stem[len(config) + 1:]  # strip "{config}-"
    return rest.split("-epoch")[0]


def _config_num_segments(checkpoint_path: Path) -> int:
    """Cheap check (no checkpoint load) -- just enough to pick the export path."""
    config = _infer_config(checkpoint_path)
    return int(REFERENCES[config].get("num_segments", 1))


def _load_checkpoint(path: Path) -> dict:
    state_dict = torch.load(path, map_location="cpu")
    return {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}


def _segment_count(state_dict: dict) -> int:
    indices = {int(key.split(".")[1]) for key in state_dict if key.startswith("models.")}
    return max(indices) + 1


def _segment_hw(state_dict: dict, segment: int) -> tuple[int, int]:
    prefix = f"models.{segment}."
    if f"{prefix}grid_features.grid" in state_dict:
        _, h, w, _ = state_dict[f"{prefix}grid_features.grid"].shape
    elif f"{prefix}real_tucker.UH.U" in state_dict:
        h = state_dict[f"{prefix}real_tucker.UH.U"].shape[0]
        w = state_dict[f"{prefix}real_tucker.UW.U"].shape[0]
    else:
        raise ValueError(
            f"Can't recover H/W for segment {segment}: no grid_features or real_tucker "
            "head in this checkpoint's config."
        )
    # H/W are stored post-downsample; this is exact only when the original
    # dims were already multiples of _K (true for any sane video resolution).
    return h * _K, w * _K


def _segment_frames(state_dict: dict, segment: int, operator_steps: int) -> int:
    prefix = f"models.{segment}."
    if f"{prefix}real_tucker.UT.U" in state_dict:
        t_padded = state_dict[f"{prefix}real_tucker.UT.U"].shape[0]
    elif f"{prefix}complex_tucker.UT.U_real" in state_dict:
        t_padded = state_dict[f"{prefix}complex_tucker.UT.U_real"].shape[0]
    else:
        raise ValueError(
            f"Can't recover T for segment {segment}: no real/complex tucker head in this "
            "checkpoint's config (grid-only configs don't bake T into any parameter)."
        )
    return t_padded - 2 * operator_steps  # undo the virtual padding NikaBlock adds


def _infer_video_shape(state_dict: dict, operator_steps: int) -> tuple[int, int, int]:
    num_segments = _segment_count(state_dict)
    h, w = _segment_hw(state_dict, 0)
    total_t = sum(_segment_frames(state_dict, i, operator_steps) for i in range(num_segments))
    return h, w, total_t


class NikaExportWrapper(nn.Module):
    """CPU float32 wrapper around MosaicNika for Kuma WebGPU export (single-segment
    configs only -- see NikaSegmentExportWrapper for multi-segment).

    Input:  norm_t — [B] float32, normalized frame times in [0, 1]
    Output:        — [B, 3, H, W] float32 RGB frames, decoding exactly the
                     requested time(s) — nothing more. B is fixed at export
                     time (whatever shape `example_input` is traced with);
                     decoding more frames means calling the model B times, not
                     baking a bigger B into the export.
    """

    def __init__(self, mosaic: MosaicNika):
        super().__init__()
        self.mosaic = mosaic

    def forward(self, norm_t: torch.Tensor) -> torch.Tensor:
        return self.mosaic(norm_t)


class NikaSegmentExportWrapper(nn.Module):
    """Wraps MosaicNika.forward_segment for one fixed segment_id (a plain Python int,
    baked in as a trace-time constant -- not a runtime input), so each segment traces
    to its own independent static graph. Used by build_branching_export to produce the
    per-branch ExportedPrograms that kuma.branching.compile_branching assembles."""

    def __init__(self, mosaic: MosaicNika, segment_id: int):
        super().__init__()
        self.mosaic = mosaic
        self.segment_id = segment_id

    def forward(self, local_norm_t: torch.Tensor) -> torch.Tensor:
        return self.mosaic.forward_segment(self.segment_id, local_norm_t)


def _load_model(checkpoint_path: Path) -> tuple[MosaicNika, int]:
    """Load a checkpoint into a MosaicNika instance. Returns (model, num_segments)."""
    config = _infer_config(checkpoint_path)
    dataset = _infer_dataset(checkpoint_path, config)
    model_kwargs = dict(REFERENCES[config])
    num_segments = model_kwargs.pop("num_segments", 1)

    # Mirrors nika.py's feature_test(): grid_ranks is doubled for the bunny
    # dataset to keep param count comparable across datasets at train time.
    if dataset == "bunny":
        model_kwargs["grid_ranks"] = model_kwargs["grid_ranks"] * 2
        print(f"[kuma_export] dataset '{dataset}': doubling grid_ranks to {model_kwargs['grid_ranks']}")

    state_dict = _load_checkpoint(checkpoint_path)
    # h, w, t describe the checkpoint's trained video shape (needed to construct
    # MosaicNika's segments correctly) -- unrelated to batch_size, which only
    # controls how many norm_t values a single-segment export is traced with.
    h, w, t = _infer_video_shape(state_dict, model_kwargs["operator_steps"])

    if num_segments > 1:
        print(f"[kuma_export] config '{config}' uses num_segments={num_segments}; total frames={t}.")

    model = MosaicNika(
        target_shape=[4, h, w, t],
        k=_K,
        model_kwargs=model_kwargs,
        out_channels=3,
        device="cpu",
        num_segments=num_segments,
    )

    mapped_sd = {
        mk: state_dict[mk.replace("._orig_mod", "")]
        for mk in model.state_dict()
        if mk.replace("._orig_mod", "") in state_dict
    }
    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    if missing:
        preview = missing[:5]
        suffix = f" … ({len(missing) - 5} more)" if len(missing) > 5 else ""
        print(f"[kuma_export] Warning: missing keys: {preview}{suffix}")

    return model.cpu(), num_segments


def build_export_inputs(
    checkpoint_path: Path, batch_size: int = 1
) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    """Single-graph export path (num_segments == 1)."""
    model, _num_segments = _load_model(checkpoint_path)
    example_input = (torch.linspace(0.0, 1.0, batch_size, dtype=torch.float32),)
    return NikaExportWrapper(model), example_input


def _playback_duration_seconds(total_frames: int, fps: float) -> float:
    """How many real seconds a full norm_t: 0->1 sweep represents, at the given fps --
    used as Kuma's optional playback metadata (see kuma's root README's .iph format
    section), so a runtime (e.g. kuma-bart's demo) plays this export back at the speed
    it was actually authored for instead of guessing one fixed speed for every export.
    Neither total_frames nor fps is recoverable from the exported graph itself; fps in
    particular isn't stored anywhere in this checkpoint's own metadata (no per-config
    fps in configs.py's REFERENCES) -- it's a real authoring choice, not a derived
    fact, so it's a CLI flag here rather than something this script infers."""
    return total_frames / fps


def _route_by_time_snippet(total_frames: int, segment_size: int, num_segments: int) -> str:
    """JS mirror of MosaicNika.make_segment_map's formula. This is the router
    js_snippet bundled into the .iph and run by kuma-bart's scheduler (CPU-side,
    before any GPU dispatch) to pick exactly one branch. Hand-written to match
    make_segment_map, not auto-derived from it -- see kuma-bart's plan notes on why
    snippets are caller-provided rather than traced. If make_segment_map's formula
    ever changes, this must be updated to match.
    """
    return f"""
function main(inputs) {{
    var normT = inputs[0];
    var totalFrames = {total_frames};
    var segmentSize = {segment_size};
    var numSegments = {num_segments};
    var segmentIds = [];
    var localNorm = [];
    for (var i = 0; i < normT.length; i++) {{
        var position = Math.min(Math.max(normT[i] * Math.max(totalFrames - 1, 1), 0.0), totalFrames - 1);
        var frameId = Math.floor(position + 1e-4);
        var segmentId = Math.min(Math.floor(frameId / segmentSize), numSegments - 1);
        var segStart = segmentId * segmentSize;
        var segEnd = Math.min((segmentId + 1) * segmentSize, totalFrames);
        var segDenom = Math.max(segEnd - segStart - 1.0, 1.0);
        var local = Math.min(Math.max((position - segStart) / segDenom, 0.0), 1.0);
        segmentIds.push(segmentId);
        localNorm.push(local);
    }}
    return [segmentIds, localNorm];
}}
""".strip()


def build_branching_export(checkpoint_path: Path, fps: float):
    """Multi-segment export path (num_segments > 1): each segment traced
    independently via forward_segment, assembled with a router snippet via
    kuma.branching.compile_branching -- kuma-bart only ever runs the one segment a
    given norm_t actually needs, not all of them."""
    model, num_segments = _load_model(checkpoint_path)
    if num_segments <= 1:
        raise ValueError("build_branching_export is only for num_segments > 1 configs")

    from kuma.export import export_program

    example_local_t = torch.tensor([0.0], dtype=torch.float32)
    branch_eps = []
    for segment_id in range(num_segments):
        wrapper = NikaSegmentExportWrapper(model, segment_id)
        branch_eps.append(export_program(wrapper, (example_local_t,)))

    snippet_source = _route_by_time_snippet(model.total_frames, model._segment_size, num_segments)
    duration_seconds = _playback_duration_seconds(model.total_frames, fps)

    from kuma.branching import compile_branching

    return compile_branching(
        router_snippet_name="route_by_time.js",
        router_snippet_source=snippet_source,
        router_input_names=["norm_t"],
        router_output_specs=[{"shape": [1], "dtype": "int64"}, {"shape": [1], "dtype": "float32"}],
        selector_output_index=0,
        branch_input_output_index=1,
        fps=fps,
        duration_seconds=duration_seconds,
        branch_eps=branch_eps,
        branch_example_inputs=[(example_local_t,)] * num_segments,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a MosaicNika checkpoint to a Kuma .iph package.")
    parser.add_argument(
        "--model", required=True, type=Path, metavar="CHECKPOINT.torch",
        help="Path to a .torch checkpoint. Config, H, W, and T are all derived from it.",
    )
    parser.add_argument(
        "--out", required=True, type=Path, metavar="FILE.iph",
        help="Where to write the self-contained .iph package.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, metavar="B",
        help="Single-segment configs only: how many norm_t values per call this export "
             "is traced/fixed for (default: 1, single-frame decode). Multi-segment "
             "configs always decode exactly the requested frame, routed at runtime.",
    )
    parser.add_argument(
        "--fps", type=float, default=24.0, metavar="FPS",
        help="Playback rate this checkpoint's frames were authored/rendered at -- not "
             "recoverable from the checkpoint itself (no per-config fps in configs.py's "
             "REFERENCES), so it's a flag, not an inferred value. Default 24 matches "
             "run_models.py's make_mp4 default, the existing convention for rendering "
             "this project's predicted videos. Embedded in the .iph as optional playback "
             "metadata (alongside the derived duration_seconds = total_frames / fps) so "
             "a runtime (e.g. kuma-bart's demo) plays this export back at the speed it "
             "was actually authored for, instead of guessing one fixed speed for every "
             "export -- different checkpoints commonly need different speeds.",
    )
    args = parser.parse_args()

    print(f"[kuma_export] checkpoint: {args.model}")

    if _config_num_segments(args.model) > 1:
        package = build_branching_export(args.model, fps=args.fps)
    else:
        model, example_input = build_export_inputs(args.model, batch_size=args.batch_size)
        duration_seconds = _playback_duration_seconds(model.mosaic.total_frames, args.fps)

        from kuma.export import export_program
        from kuma.compiler import compile as kuma_compile

        ep = export_program(model, example_input)
        package = kuma_compile(ep, fps=args.fps, duration_seconds=duration_seconds)

    total_bytes = sum(w["byte_length"] for w in package.manifest["weights"])
    print(
        f"[kuma_export] {len(package.manifest['weights'])} weight tensors, {total_bytes:,} bytes; "
        f"{package.manifest['graph']['node_count']} graph nodes"
    )
    playback = package.manifest.get("playback")
    if playback:
        print(f"[kuma_export] playback: {playback['duration_seconds']:.2f}s/sweep @ {playback['fps']}fps")

    package.save(args.out)
    print(f"[kuma_export] wrote {args.out.resolve()} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
