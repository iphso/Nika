REFERENCES = {
    "xxs": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 40, 40, 40],
        "complex_tucker_ranks": [2, 30, 30, 30],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
    },
    "xs": {
        "grid_ranks": 2,
        "real_tucker_ranks": [3, 50, 50, 50],
        "complex_tucker_ranks": [3, 40, 40, 40],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
    },
    "small": {
        "grid_ranks": 4,
        "real_tucker_ranks": [3, 75, 75, 60],
        "complex_tucker_ranks": [3, 60, 60, 40],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 1,
    },
    "medium": {
        "grid_ranks": 8,
        "real_tucker_ranks": [3, 90, 90, 70],
        "complex_tucker_ranks": [3, 75, 75, 60],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 1,
    },
    "large": {
        "grid_ranks": 16,
        "real_tucker_ranks": [4, 100, 100, 100],
        "complex_tucker_ranks": [4, 80, 80, 80],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 1,
    },
    "xs2": {
        "grid_ranks": 2,
        "real_tucker_ranks": [3, 50, 50, 50],
        "complex_tucker_ranks": [3, 40, 40, 40],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 2,
    },
    "xs4": {
        "grid_ranks": 2,
        "real_tucker_ranks": [3, 50, 50, 50],
        "complex_tucker_ranks": [3, 40, 40, 40],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xs8": {
        "grid_ranks": 2,
        "real_tucker_ranks": [3, 50, 50, 50],
        "complex_tucker_ranks": [3, 40, 40, 40],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 8,
    },
    "xxs1": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 40, 40, 40],
        "complex_tucker_ranks": [2, 30, 30, 30],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 1,
    },
    "xxs2": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 40, 40, 40],
        "complex_tucker_ranks": [2, 30, 30, 30],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 2,
    },
    "xxs4": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 40, 40, 40],
        "complex_tucker_ranks": [2, 30, 30, 30],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs8": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 40, 40, 40],
        "complex_tucker_ranks": [2, 30, 30, 30],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 8,
    },
    "xxs16": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 40, 40, 40],
        "complex_tucker_ranks": [2, 30, 30, 30],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 16,
    },
    "xxs4-noop": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 40, 40, 40],
        "complex_tucker_ranks": [2, 30, 30, 30],
        "conv_hidden": 48,
        "operator_steps": 0,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs4-real": {
        "grid_ranks": 4,
        "real_tucker_ranks": [2, 50, 50, 50],
        "complex_tucker_ranks": False,
        "complex_grid_ranks": False,
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs4-complex": {
        "grid_ranks": False,
        "real_tucker_ranks": False,
        "complex_tucker_ranks": [3, 40, 40, 40],
        "complex_grid_ranks": 4,
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs4-tucker": {
        "grid_ranks": False,
        "real_tucker_ranks": [2, 50, 50, 50],
        "complex_tucker_ranks": [3, 40, 40, 40],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs4-grid": {
        "grid_ranks": 3,
        "real_tucker_ranks": False,
        "complex_tucker_ranks": False,
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    # --- Tucker rank-allocation exploration (appendix) ---
    # All variants below are iso-param with "xxs2" (rC fixed at 2, same grid/conv/operator
    # settings, num_segments=2) to isolate the effect of *how* rank is split across
    # (rH, rW, rT) rather than total budget. Matched to within ~1% of xxs2's tucker-head
    # param count (real=150720, complex=132540 on bunny's internal [C4,H180,W320,T68] shape).
    "xxs2-wbig": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 32, 62, 31],
        "complex_tucker_ranks": [2, 26, 48, 21],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 2,
    },
    "xxs2-tbig": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 33, 30, 66],
        "complex_tucker_ranks": [2, 22, 24, 52],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 2,
    },
    "xxs2-hbig": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 66, 33, 29],
        "complex_tucker_ranks": [2, 52, 20, 25],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 2,
    },
    "xxs2-hwbig-tsmall": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 62, 57, 17],
        "complex_tucker_ranks": [2, 44, 44, 13],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 2,
    },
    "xxs2-tbig-hwsmall": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 29, 26, 88],
        "complex_tucker_ranks": [2, 20, 19, 72],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 2,
    },
    "xxs2-wbig-ext": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 20, 135, 19],
        "complex_tucker_ranks": [2, 15, 93, 17],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 2,
    },
    # Same rank-allocation variants, run on the xxs4 tier (num_segments=4, ~3M params)
    # instead of xxs2. The Tucker core tensor (rC*rH*rW*rT) dominates head param count and
    # depends only on the rank tuple, not on num_segments/target T, so these are iso-param
    # with xxs4 the same way the xxs2-* variants are iso-param with xxs2.
    "xxs4-wbig": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 32, 62, 31],
        "complex_tucker_ranks": [2, 26, 48, 21],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs4-tbig": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 33, 30, 66],
        "complex_tucker_ranks": [2, 22, 24, 52],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs4-hbig": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 66, 33, 29],
        "complex_tucker_ranks": [2, 52, 20, 25],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs4-hwbig-tsmall": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 62, 57, 17],
        "complex_tucker_ranks": [2, 44, 44, 13],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs4-tbig-hwsmall": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 29, 26, 88],
        "complex_tucker_ranks": [2, 20, 19, 72],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
    "xxs4-wbig-ext": {
        "grid_ranks": 2,
        "real_tucker_ranks": [2, 20, 135, 19],
        "complex_tucker_ranks": [2, 15, 93, 17],
        "conv_hidden": 48,
        "operator_steps": 1,
        "op_hdim": 48,
        "num_segments": 4,
    },
}

# UVG frames are 2.25x Bunny's area, so the xxs Tucker factors grow and xxsN overshoots the
# param ladder by ~20% there; spatial ranks 40/30 -> 32/24 bring xxs1..xxs16 to 0.75..12M (+-6%).
for _n in (1, 2, 4, 8, 16):
    REFERENCES[f"xxs{_n}-r32"] = dict(REFERENCES[f"xxs{_n}"], real_tucker_ranks=[2, 32, 32, 32],
                                      complex_tucker_ranks=[2, 24, 24, 24])

# Bench conv-width probe: the ConvNeXt upres head runs at full resolution on every frame but holds
# few params, so its width trades encode/decode speed against fidelity at ~constant model size.
for _base in [k for k in REFERENCES if k.startswith("xxs") and k[3:].split("-")[0].isdigit()
              and (k.count("-") == 0 or k.endswith("-r32"))]:
    for _w in (16, 24, 32, 64, 96):
        REFERENCES[f"{_base}-c{_w}"] = dict(REFERENCES[_base], conv_hidden=_w)

# ---------------------------------------------------------------------------------------------
# Camera-ready ablation suite (2026-11-05). Everything below is built on the benchmark's Bunny 3M
# config (xxs4: segmented, 4 segments) or, for the UVG confirmation, UVG 3M (xs2). Segmentation is
# never changed. Bunny doubles grid_ranks at build time (feature_test); explicit *_grid_ranks of the
# new head types below are absolute channel counts and are NOT doubled.
# ---------------------------------------------------------------------------------------------

def _scale_latent(base, s):
    """Latent capacity x s: every latent family's parameters scale by ~s, relative allocation kept.
    Grid channels scale linearly (exact); Tucker cores are rC*rH*rW*rT with rC fixed (C=4), so the
    spatial/temporal ranks scale by s^(1/3). Decoder and temporal operators untouched."""
    out = dict(base)
    out["grid_ranks"] = max(1, round(base["grid_ranks"] * s))
    for key in ("real_tucker_ranks", "complex_tucker_ranks"):
        r = base[key]
        out[key] = [r[0]] + [max(1, round(x * s ** (1 / 3))) for x in r[1:]]
    return out


def _scale_decoder(base, s):
    """Decoder capacity x s: ConvNeXt hidden width only (one block, as in every config)."""
    return dict(base, conv_hidden=int(round(base["conv_hidden"] * s)))


for _base_name in ("xxs4", "xs2"):
    for _l in (0.5, 1, 2):
        for _d in (0.5, 1, 2):
            REFERENCES[f"{_base_name}-L{_l:g}-D{_d:g}"] = _scale_decoder(_scale_latent(REFERENCES[_base_name], _l), _d)

# Temporal operator capacity (Experiment C): FiLM conv operator hidden width; "steps2" uses the
# implementation's operator_steps=2, i.e. a +-2-frame window with a second forward/backward operator
# pair whose residuals are summed with the first pair's (parallel, not sequential refinement).
REFERENCES["xxs4-op24"] = dict(REFERENCES["xxs4"], op_hdim=24)
REFERENCES["xxs4-op96"] = dict(REFERENCES["xxs4"], op_hdim=96)
REFERENCES["xxs4-steps2"] = dict(REFERENCES["xxs4"], operator_steps=2)

# Homogeneous-composition controls (Experiment D), Bunny only (absolute grid channels).
# dupreal: the spectral head (complex Tucker x complex grid -> irfft2) is replaced by its real-domain
#   mirror, an independently parameterized real Tucker x real grid head with no FFT. Same head count
#   (latent output 3C channels), same grid footprint (4 x 180 x 320 vs 8 x 180 x 161), Tucker rank set
#   so the head's parameters match the spectral head's (~362k per segment).
REFERENCES["xxs4-dupreal"] = dict(REFERENCES["xxs4"], complex_tucker_ranks=False, complex_grid_ranks=False,
                                  realprod_tucker_ranks=[2, 38, 38, 38], realprod_grid_ranks=4)
# dupspec: the two real heads (real Tucker, real grid) are replaced by their spectral mirrors, a
#   complex-Tucker-only head and a complex-grid-only head (each through irfft2), next to the standard
#   spectral head. complex_grid_ranks is pinned to the standard Bunny value (4) because it would
#   otherwise inherit grid_ranks=False.
REFERENCES["xxs4-dupspec"] = dict(REFERENCES["xxs4"], real_tucker_ranks=False, grid_ranks=False, complex_grid_ranks=4,
                                  spec_tucker_ranks=[2, 32, 32, 32], spec_grid_ranks=4)
