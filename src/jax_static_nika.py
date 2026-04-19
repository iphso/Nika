from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@dataclass(frozen=True)
class StaticNikaSpec:
    channels: int
    height: int
    width: int
    frames: int
    upscale: int = 4
    num_groups: int = 3
    output_channels: int = 3


def load_reference_params(path: str | Path) -> dict[str, jnp.ndarray]:
    with np.load(path) as data:
        return {key: jnp.asarray(data[key], dtype=jnp.float32) for key in data.files}


def infer_spec(params: dict[str, jnp.ndarray]) -> StaticNikaSpec:
    channels = int(params["real_tucker.UC.U"].shape[0])
    height = int(params["real_tucker.UH.U"].shape[0])
    width = int(params["real_tucker.UW.U"].shape[0])
    frames = int(params["real_tucker.UT.U"].shape[0])
    output_channels = int(params["upres.layers.8.weight"].shape[0] // 16)
    return StaticNikaSpec(
        channels=channels,
        height=height,
        width=width,
        frames=frames,
        output_channels=output_channels,
    )


def _col_norm(matrix: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    norms = jnp.sqrt(jnp.sum(jnp.real(matrix * jnp.conj(matrix)), axis=0, keepdims=True) + eps)
    return matrix / norms


def _tucker_construct(
    ut: jnp.ndarray,
    uc: jnp.ndarray,
    uh: jnp.ndarray,
    uw: jnp.ndarray,
    core: jnp.ndarray,
) -> jnp.ndarray:
    ut = _col_norm(ut)
    uc = _col_norm(uc)
    uh = _col_norm(uh)
    uw = _col_norm(uw)
    x = jnp.tensordot(core, ut, axes=([3], [1]))
    x = jnp.transpose(x, (3, 2, 0, 1))
    x = jnp.tensordot(uc, x, axes=([1], [1]))
    x = jnp.transpose(x, (1, 0, 2, 3))
    x = jnp.tensordot(uh, x, axes=([1], [2]))
    x = jnp.transpose(x, (1, 2, 0, 3))
    x = jnp.tensordot(uw, x, axes=([1], [3]))
    return jnp.transpose(x, (1, 2, 3, 0))


def _hermitian_project_shifted(freqs: jnp.ndarray) -> jnp.ndarray:
    ku = jnp.fft.ifftshift(freqs, axes=(-2, -1))
    partner = jnp.roll(
        jnp.roll(jnp.flip(ku, axis=(-2, -1)), shift=1, axis=-2),
        shift=1,
        axis=-1,
    ).conj()
    ksym = 0.5 * (ku + partner)
    ksym = ksym.at[..., 0, 0].set(jnp.real(ksym[..., 0, 0]) + 0j)
    kw = int(ksym.shape[-1])
    if kw % 2 == 0:
        mid = kw // 2
        ksym = ksym.at[..., 0, mid].set(jnp.real(ksym[..., 0, mid]) + 0j)
        ksym = ksym.at[..., mid, 0].set(jnp.real(ksym[..., mid, 0]) + 0j)
        ksym = ksym.at[..., mid, mid].set(jnp.real(ksym[..., mid, mid]) + 0j)
    return jnp.fft.fftshift(ksym, axes=(-2, -1))


def real_tucker(params: dict[str, jnp.ndarray], frame_idx: jnp.ndarray) -> jnp.ndarray:
    ut = params["real_tucker.UT.U"][frame_idx]
    uc = params["real_tucker.UC.U"]
    uh = params["real_tucker.UH.U"]
    uw = params["real_tucker.UW.U"]
    core = params["real_tucker.G"]
    return _tucker_construct(ut, uc, uh, uw, core)


def complex_tucker(params: dict[str, jnp.ndarray], frame_idx: jnp.ndarray) -> jnp.ndarray:
    ut = params["complex_tucker.UT.U_real"][frame_idx] + 1j * params["complex_tucker.UT.U_imag"][frame_idx]
    uc = params["complex_tucker.UC.U_real"] + 1j * params["complex_tucker.UC.U_imag"]
    uh = params["complex_tucker.UH.U_real"] + 1j * params["complex_tucker.UH.U_imag"]
    uw = params["complex_tucker.UW.U_real"] + 1j * params["complex_tucker.UW.U_imag"]
    core = params["complex_tucker.G_real"] + 1j * params["complex_tucker.G_imag"]
    constructed = _tucker_construct(ut, uc, uh, uw, core)
    projected = _hermitian_project_shifted(constructed)
    return jnp.fft.ifft2(projected, axes=(-2, -1), norm="ortho").real.astype(jnp.float32)


def feature_grid(
    params: dict[str, jnp.ndarray],
    norm_t: jnp.ndarray,
    spec: StaticNikaSpec,
) -> jnp.ndarray:
    grid = params["grid_features.grid"]
    grid_c, grid_h, grid_w, grid_t = [int(v) for v in grid.shape]

    z = jnp.clip(norm_t, 0.0, 1.0) * float(grid_t - 1)
    z0 = jnp.floor(z).astype(jnp.int32)
    z1 = jnp.minimum(z0 + 1, grid_t - 1)
    wz = (z - z0.astype(jnp.float32)).reshape(-1, 1, 1, 1)

    g0 = jnp.moveaxis(jnp.take(grid, z0, axis=3), -1, 0)
    g1 = jnp.moveaxis(jnp.take(grid, z1, axis=3), -1, 0)
    sampled_t = g0 * (1.0 - wz) + g1 * wz

    y = jnp.linspace(0.0, float(grid_h - 1), spec.height, dtype=jnp.float32)
    x = jnp.linspace(0.0, float(grid_w - 1), spec.width, dtype=jnp.float32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, grid_h - 1)
    x1 = jnp.minimum(x0 + 1, grid_w - 1)
    wy = (y - y0.astype(jnp.float32)).reshape(1, 1, spec.height, 1)
    wx = (x - x0.astype(jnp.float32)).reshape(1, 1, 1, spec.width)

    sy0 = jnp.take(sampled_t, y0, axis=2)
    sy1 = jnp.take(sampled_t, y1, axis=2)
    s00 = jnp.take(sy0, x0, axis=3)
    s01 = jnp.take(sy0, x1, axis=3)
    s10 = jnp.take(sy1, x0, axis=3)
    s11 = jnp.take(sy1, x1, axis=3)

    sx0 = s00 * (1.0 - wx) + s01 * wx
    sx1 = s10 * (1.0 - wx) + s11 * wx
    sampled = sx0 * (1.0 - wy) + sx1 * wy

    proj_w = params["grid_features.channel_proj.weight"]
    proj_b = params["grid_features.channel_proj.bias"]
    sampled_hwc = jnp.transpose(sampled, (0, 2, 3, 1))
    projected = jnp.einsum("bhwc,oc->bhwo", sampled_hwc, proj_w) + proj_b
    return jnp.transpose(projected, (0, 3, 1, 2))


def group_norm(params: dict[str, jnp.ndarray], x: jnp.ndarray, spec: StaticNikaSpec) -> jnp.ndarray:
    batch, channels, height, width = x.shape
    group_size = channels // spec.num_groups
    reshaped = x.reshape(batch, spec.num_groups, group_size, height, width)
    mean = reshaped.mean(axis=(2, 3, 4), keepdims=True)
    var = reshaped.var(axis=(2, 3, 4), keepdims=True)
    normalized = (reshaped - mean) / jnp.sqrt(var + 1e-5)
    normalized = normalized.reshape(batch, channels, height, width)
    weight = params["groupnorm.weight"].reshape(1, channels, 1, 1)
    bias = params["groupnorm.bias"].reshape(1, channels, 1, 1)
    return normalized * weight + bias


def _conv2d_nchw(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    padding: tuple[tuple[int, int], tuple[int, int]] | str,
    groups: int = 1,
) -> jnp.ndarray:
    y = lax.conv_general_dilated(
        lhs=x,
        rhs=weight,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=groups,
    )
    return y + bias.reshape(1, -1, 1, 1)


def _silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


def _pixel_shuffle(x: jnp.ndarray, upscale: int) -> jnp.ndarray:
    batch, channels, height, width = x.shape
    out_channels = channels // (upscale * upscale)
    reshaped = x.reshape(batch, out_channels, upscale, upscale, height, width)
    transposed = reshaped.transpose(0, 1, 4, 2, 5, 3)
    return transposed.reshape(batch, out_channels, height * upscale, width * upscale)


def basic_upres(params: dict[str, jnp.ndarray], x: jnp.ndarray, spec: StaticNikaSpec) -> jnp.ndarray:
    x = _conv2d_nchw(
        x,
        params["upres.layers.0.weight"],
        params["upres.layers.0.bias"],
        padding=((1, 1), (1, 1)),
    )
    x = _silu(x)
    x = _conv2d_nchw(
        x,
        params["upres.layers.2.weight"],
        params["upres.layers.2.bias"],
        padding=((1, 1), (1, 1)),
        groups=int(x.shape[1]),
    )
    x = _silu(x)
    x = _conv2d_nchw(
        x,
        params["upres.layers.4.weight"],
        params["upres.layers.4.bias"],
        padding="VALID",
    )
    x = _silu(x)
    x = _conv2d_nchw(
        x,
        params["upres.layers.6.weight"],
        params["upres.layers.6.bias"],
        padding=((1, 1), (1, 1)),
        groups=int(x.shape[1]),
    )
    x = _silu(x)
    x = _conv2d_nchw(
        x,
        params["upres.layers.8.weight"],
        params["upres.layers.8.bias"],
        padding="VALID",
    )
    return _pixel_shuffle(x, spec.upscale)


def forward_with_intermediates(
    params: dict[str, jnp.ndarray],
    frame_idx: jnp.ndarray,
    spec: StaticNikaSpec,
) -> dict[str, jnp.ndarray]:
    frame_idx = jnp.asarray(frame_idx, dtype=jnp.int32).reshape(-1)
    norm_t = frame_idx.astype(jnp.float32) / float(spec.frames - 1)
    grid_out = feature_grid(params, norm_t, spec)
    real_out = real_tucker(params, frame_idx)
    complex_out = complex_tucker(params, frame_idx)
    base_input = jnp.concatenate([grid_out, real_out, complex_out], axis=1)
    normed = group_norm(params, base_input, spec)
    output = basic_upres(params, normed, spec)
    return {
        "t": frame_idx.astype(jnp.float32),
        "grid_features": grid_out,
        "real_tucker": real_out,
        "complex_tucker": complex_out,
        "base_input": base_input,
        "groupnorm": normed,
        "output": output,
    }


def forward(
    params: dict[str, jnp.ndarray],
    frame_idx: jnp.ndarray,
    spec: StaticNikaSpec,
) -> jnp.ndarray:
    return forward_with_intermediates(params, frame_idx, spec)["output"]
