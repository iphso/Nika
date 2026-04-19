from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@dataclass(frozen=True)
class CurrentNikaSpec:
    channels: int
    height: int
    width: int
    frames: int
    n_heads: int
    operator_steps: int
    window_size: int
    upscale: int
    output_channels: int
    encoding_dim: int


def load_params(path: str | Path) -> dict[str, jnp.ndarray]:
    with np.load(path) as data:
        return {key: jnp.asarray(data[key], dtype=jnp.float32) for key in data.files}


def infer_spec(params: dict[str, jnp.ndarray]) -> CurrentNikaSpec:
    channels = int(params["real_tucker.UC.U"].shape[0])
    height = int(params["real_tucker.UH.U"].shape[0])
    width = int(params["real_tucker.UW.U"].shape[0])
    frames = int(params["real_tucker.UT.U"].shape[0])
    n_heads = int(params["groupnorm.weight"].shape[0] // channels)
    window_size = int(params["flow_operator.operator_head.0.weight"].shape[1] // (n_heads * channels))
    operator_steps = (window_size - 1) // 2
    final_channels = int(params["upres.upres.6.weight"].shape[0])
    output_channels = 3
    upscale = int(round((final_channels / output_channels) ** 0.5))
    encoding_dim = int(params["flow_operator.t_modulator.0.weight"].shape[1])
    return CurrentNikaSpec(
        channels=channels,
        height=height,
        width=width,
        frames=frames,
        n_heads=n_heads,
        operator_steps=operator_steps,
        window_size=window_size,
        upscale=upscale,
        output_channels=output_channels,
        encoding_dim=encoding_dim,
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
    x = jnp.tensordot(uc, core, axes=([1], [1]))
    x = jnp.tensordot(ut, x, axes=([1], [1]))
    x = jnp.tensordot(uh, x, axes=([1], [2]))
    x = jnp.tensordot(uw, x, axes=([1], [3]))
    return jnp.transpose(x, (2, 3, 1, 0))


def _tucker_basis(
    uc: jnp.ndarray,
    uh: jnp.ndarray,
    uw: jnp.ndarray,
    core: jnp.ndarray,
) -> jnp.ndarray:
    uc = _col_norm(uc)
    uh = _col_norm(uh)
    uw = _col_norm(uw)
    return jnp.einsum("tcab,xc,ya,zb->txyz", core, uc, uh, uw)


def _sample_grid_positions(out_size: int, src_size: int) -> jnp.ndarray:
    coords = jnp.arange(out_size, dtype=jnp.float32) * (float(src_size) / float(out_size - 1)) - 0.5
    return jnp.clip(coords, 0.0, float(src_size - 1))


def _sample_feature_grid(grid_4d: jnp.ndarray, out_h: int, out_w: int) -> jnp.ndarray:
    batch, _channels, src_h, src_w = grid_4d.shape
    y = _sample_grid_positions(out_h, int(src_h))
    x = _sample_grid_positions(out_w, int(src_w))
    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, src_h - 1)
    x1 = jnp.minimum(x0 + 1, src_w - 1)
    wy = (y - y0.astype(jnp.float32)).reshape(1, 1, out_h, 1)
    wx = (x - x0.astype(jnp.float32)).reshape(1, 1, 1, out_w)

    sy0 = jnp.take(grid_4d, y0, axis=2)
    sy1 = jnp.take(grid_4d, y1, axis=2)
    s00 = jnp.take(sy0, x0, axis=3)
    s01 = jnp.take(sy0, x1, axis=3)
    s10 = jnp.take(sy1, x0, axis=3)
    s11 = jnp.take(sy1, x1, axis=3)

    sx0 = s00 * (1.0 - wx) + s01 * wx
    sx1 = s10 * (1.0 - wx) + s11 * wx
    return sx0 * (1.0 - wy) + sx1 * wy


def _window_indices(norm_t: jnp.ndarray, spec: CurrentNikaSpec) -> jnp.ndarray:
    norm_t = jnp.asarray(norm_t, dtype=jnp.float32).reshape(-1)
    if norm_t.shape[0] != 1:
        raise ValueError("Current JAX decode path expects a single normalized timestamp")
    center_idx = jnp.floor(norm_t[0] * spec.frames).astype(jnp.int32)
    center_idx = jnp.clip(center_idx, 0, spec.frames - 1)
    offsets = jnp.arange(-spec.operator_steps, spec.operator_steps + 1, dtype=jnp.int32)
    return jnp.clip(center_idx + offsets, 0, spec.frames - 1)


def feature_grid(
    grid: jnp.ndarray,
    proj_weight: jnp.ndarray | None,
    proj_bias: jnp.ndarray | None,
    batch: int,
    out_h: int,
    out_w: int,
) -> jnp.ndarray:
    grid_4d = jnp.repeat(grid, batch, axis=0)
    if proj_weight is not None and proj_bias is not None:
        projected = jnp.einsum("bhwc,oc->bhwo", jnp.transpose(grid_4d, (0, 2, 3, 1)), proj_weight) + proj_bias
        grid_4d = jnp.transpose(projected, (0, 3, 1, 2))
    return _sample_feature_grid(grid_4d, out_h, out_w)


def prepare_runtime(params: dict[str, jnp.ndarray], spec: CurrentNikaSpec) -> dict[str, jnp.ndarray]:
    runtime = dict(params)
    runtime["__cache.real_basis"] = _tucker_basis(
        params["real_tucker.UC.U"],
        params["real_tucker.UH.U"],
        params["real_tucker.UW.U"],
        params["real_tucker.G"],
    )
    runtime["__cache.complex_basis"] = _tucker_basis(
        params["complex_tucker.UC.U_real"] + 1j * params["complex_tucker.UC.U_imag"],
        params["complex_tucker.UH.U_real"] + 1j * params["complex_tucker.UH.U_imag"],
        params["complex_tucker.UW.U_real"] + 1j * params["complex_tucker.UW.U_imag"],
        params["complex_tucker.G_real"] + 1j * params["complex_tucker.G_imag"],
    )
    runtime["__cache.grid_features"] = feature_grid(
        params["grid_features.grid"],
        None,
        None,
        spec.window_size,
        spec.height,
        spec.width,
    )
    runtime["__cache.complex_grid"] = complex_grid(runtime, spec.window_size, spec)
    return runtime


def _ensure_prepared(runtime: dict[str, jnp.ndarray], spec: CurrentNikaSpec) -> dict[str, jnp.ndarray]:
    if "__cache.real_basis" in runtime:
        return runtime
    return prepare_runtime(runtime, spec)


def real_tucker(
    runtime: dict[str, jnp.ndarray],
    window_indices: jnp.ndarray,
    spec: CurrentNikaSpec,
) -> jnp.ndarray:
    del spec
    ut = _col_norm(jnp.take(runtime["real_tucker.UT.U"], window_indices, axis=0))
    return jnp.tensordot(ut, runtime["__cache.real_basis"], axes=([1], [0]))


def complex_tucker_construct(
    runtime: dict[str, jnp.ndarray],
    window_indices: jnp.ndarray,
    spec: CurrentNikaSpec,
) -> jnp.ndarray:
    del spec
    ut = _col_norm(
        jnp.take(
            runtime["complex_tucker.UT.U_real"] + 1j * runtime["complex_tucker.UT.U_imag"],
            window_indices,
            axis=0,
        )
    )
    return jnp.tensordot(ut, runtime["__cache.complex_basis"], axes=([1], [0]))


def complex_grid(params: dict[str, jnp.ndarray], batch: int, spec: CurrentNikaSpec) -> jnp.ndarray:
    grid = feature_grid(
        params["complex_tucker.feature_grid.grid"],
        params["complex_tucker.feature_grid.channel_proj.weight"],
        params["complex_tucker.feature_grid.channel_proj.bias"],
        batch,
        spec.height,
        (spec.width // 2) + 1,
    )
    real, imag = jnp.split(grid, 2, axis=1)
    return real + 1j * imag


def complex_tucker(
    runtime: dict[str, jnp.ndarray],
    window_indices: jnp.ndarray,
    spec: CurrentNikaSpec,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    construct = complex_tucker_construct(runtime, window_indices, spec)
    grid = runtime["__cache.complex_grid"]
    multiplied = construct * grid
    return construct, grid, jnp.fft.irfft2(multiplied, s=(spec.height, spec.width), axes=(-2, -1), norm="ortho").real.astype(jnp.float32)


def group_norm(params: dict[str, jnp.ndarray], x: jnp.ndarray, spec: CurrentNikaSpec) -> jnp.ndarray:
    batch, channels, height, width = x.shape
    group_size = channels // spec.n_heads
    reshaped = x.reshape(batch, spec.n_heads, group_size, height, width)
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


def _gelu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.gelu(x, approximate=False)


def _pixel_shuffle(x: jnp.ndarray, upscale: int) -> jnp.ndarray:
    batch, channels, height, width = x.shape
    out_channels = channels // (upscale * upscale)
    reshaped = x.reshape(batch, out_channels, upscale, upscale, height, width)
    transposed = reshaped.transpose(0, 1, 4, 2, 5, 3)
    return transposed.reshape(batch, out_channels, height * upscale, width * upscale)


def fourier_encoding(freqs: jnp.ndarray, x: jnp.ndarray, target_dim: int) -> jnp.ndarray:
    x = x.reshape(-1, 1)
    projected = x[..., None] * freqs
    emb = jnp.concatenate([x[..., None], jnp.cos(projected), jnp.sin(projected)], axis=-1)
    trunc = emb[..., :target_dim]
    if trunc.ndim > 2 and trunc.shape[1] == 1:
        trunc = trunc.squeeze(1)
    return trunc


def linear(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight.T + bias


def flow_operator(
    params: dict[str, jnp.ndarray],
    op_input: jnp.ndarray,
    norm_t: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    head0 = _conv2d_nchw(op_input, params["flow_operator.operator_head.0.weight"], params["flow_operator.operator_head.0.bias"], padding="VALID")
    head1 = _gelu(head0)
    head2 = _conv2d_nchw(
        head1,
        params["flow_operator.operator_head.2.weight"],
        params["flow_operator.operator_head.2.bias"],
        padding=((1, 1), (1, 1)),
        groups=int(head1.shape[1]),
    )
    head3 = _gelu(head2)
    initial = _conv2d_nchw(head3, params["flow_operator.operator_head.4.weight"], params["flow_operator.operator_head.4.bias"], padding="VALID")

    time_emb = fourier_encoding(params["flow_operator.encoding.freqs"], norm_t, int(params["flow_operator.t_modulator.0.weight"].shape[1]))
    mod0 = linear(time_emb, params["flow_operator.t_modulator.0.weight"], params["flow_operator.t_modulator.0.bias"])
    mod1 = _gelu(mod0)
    modulation = linear(mod1, params["flow_operator.t_modulator.2.weight"], params["flow_operator.t_modulator.2.bias"])
    gamma, beta = jnp.split(modulation, 2, axis=-1)
    gamma = gamma.reshape(-1, initial.shape[1], 1, 1)
    beta = beta.reshape(-1, initial.shape[1], 1, 1)
    modulated = initial * (1.0 + gamma) + beta

    tail0 = _conv2d_nchw(
        modulated,
        params["flow_operator.operator_tail.0.weight"],
        params["flow_operator.operator_tail.0.bias"],
        padding=((1, 1), (1, 1)),
        groups=int(modulated.shape[1]),
    )
    tail1 = _gelu(tail0)
    output = _conv2d_nchw(tail1, params["flow_operator.operator_tail.2.weight"], params["flow_operator.operator_tail.2.bias"], padding="VALID")

    return {
        "time_emb": time_emb,
        "gamma": gamma,
        "beta": beta,
        "initial": initial,
        "output": output,
    }


def basic_upres(params: dict[str, jnp.ndarray], x: jnp.ndarray, spec: CurrentNikaSpec) -> jnp.ndarray:
    x = _conv2d_nchw(x, params["upres.upres.0.weight"], params["upres.upres.0.bias"], padding="VALID")
    x = _gelu(x)
    x = _conv2d_nchw(
        x,
        params["upres.upres.2.weight"],
        params["upres.upres.2.bias"],
        padding=((1, 1), (1, 1)),
        groups=int(x.shape[1]),
    )
    x = _gelu(x)
    x = _conv2d_nchw(x, params["upres.upres.4.weight"], params["upres.upres.4.bias"], padding="VALID")
    x = _gelu(x)
    x = _conv2d_nchw(x, params["upres.upres.6.weight"], params["upres.upres.6.bias"], padding="VALID")
    return _pixel_shuffle(x, spec.upscale)


def forward_with_intermediates(
    params: dict[str, jnp.ndarray],
    norm_t: jnp.ndarray,
    spec: CurrentNikaSpec,
) -> dict[str, jnp.ndarray]:
    runtime = _ensure_prepared(params, spec)
    norm_t = jnp.asarray(norm_t, dtype=jnp.float32).reshape(-1)
    window_indices = _window_indices(norm_t, spec)

    real_out = real_tucker(runtime, window_indices, spec)
    grid_out = runtime["__cache.grid_features"]
    complex_construct, complex_grid_out, complex_out = complex_tucker(runtime, window_indices, spec)
    response_input = jnp.concatenate([real_out, grid_out, complex_out], axis=1)
    response = group_norm(runtime, response_input, spec)

    aggregated = response[spec.operator_steps]
    op_input = response.reshape(1, -1, spec.height, spec.width)
    op = flow_operator(runtime, op_input, norm_t)
    aggregated = aggregated[None, ...] + op["output"]
    output = basic_upres(runtime, aggregated, spec)

    return {
        "norm_t": norm_t,
        "real_tucker": real_out,
        "grid_features": grid_out,
        "complex_tucker_construct": complex_construct,
        "complex_tucker_grid": complex_grid_out,
        "complex_tucker": complex_out,
        "response_input": response_input,
        "groupnorm": response,
        "operator_input": op_input,
        "operator_initial": op["initial"],
        "operator_time_emb": op["time_emb"],
        "operator_gamma": op["gamma"],
        "operator_beta": op["beta"],
        "operator_output": op["output"],
        "aggregated": aggregated,
        "output": output,
    }


def forward(params: dict[str, jnp.ndarray], norm_t: jnp.ndarray, spec: CurrentNikaSpec) -> jnp.ndarray:
    return forward_with_intermediates(params, norm_t, spec)["output"]


def batch_forward(
    params: dict[str, jnp.ndarray],
    norm_t_batch: jnp.ndarray,
    spec: CurrentNikaSpec,
) -> jnp.ndarray:
    runtime = _ensure_prepared(params, spec)
    norm_t_batch = jnp.asarray(norm_t_batch, dtype=jnp.float32).reshape(-1)
    return jax.vmap(lambda nt: forward(runtime, jnp.asarray([nt], dtype=jnp.float32), spec)[0])(norm_t_batch)


def batch_psnr_loss(
    params: dict[str, jnp.ndarray],
    norm_t_batch: jnp.ndarray,
    targets: jnp.ndarray,
    spec: CurrentNikaSpec,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    predictions = batch_forward(params, norm_t_batch, spec)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    mse = jnp.mean((predictions - targets) ** 2, axis=(1, 2, 3))
    psnr = -10.0 * jnp.log10(mse + 1e-8)
    loss = jnp.mean(-psnr)
    return loss, {
        "predictions": predictions,
        "mse": mse,
        "psnr": psnr,
    }
