from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class SOAPHyperParams:
    lr: float = 3e-3
    betas: tuple[float, float] = (0.95, 0.95)
    shampoo_beta: float = -1.0
    eps: float = 1e-8
    weight_decay: float = 0.01
    precondition_frequency: int = 10
    max_precond_dim: int = 10000
    merge_dims: bool = False
    precondition_1d: bool = False
    normalize_grads: bool = False
    data_format: str = "channels_first"
    correct_bias: bool = True


def _lerp(current: jnp.ndarray, target: jnp.ndarray, weight: float) -> jnp.ndarray:
    return current + weight * (target - current)


def merge_dims(grad: jnp.ndarray, max_precond_dim: int, data_format: str = "channels_first") -> jnp.ndarray:
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unsupported data format: {data_format}")
    if data_format == "channels_last" and grad.ndim == 4:
        grad = jnp.transpose(grad, (0, 3, 1, 2))

    new_shape: list[int] = []
    curr_shape = 1
    for sh in grad.shape:
        temp_shape = curr_shape * int(sh)
        if temp_shape > max_precond_dim:
            if curr_shape > 1:
                new_shape.append(curr_shape)
                curr_shape = int(sh)
            else:
                new_shape.append(int(sh))
                curr_shape = 1
        else:
            curr_shape = temp_shape

    if curr_shape > 1 or not new_shape:
        new_shape.append(curr_shape)

    return grad.reshape(tuple(new_shape))


def _empty_like_torch_skip() -> None:
    return None


def _is_active_matrix(matrix: jnp.ndarray | None) -> bool:
    return matrix is not None


def _canonicalize_column_signs(mat: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    if mat.size == 0:
        return mat
    idx = jnp.argmax(jnp.abs(mat), axis=0)
    signs = jnp.sign(mat[idx, jnp.arange(mat.shape[1])])
    signs = jnp.where(signs == 0, 1.0, signs)
    return mat * signs.reshape(1, -1)


def _canonicalize_subspace(basis: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    if basis.size == 0:
        return basis
    dim, width = basis.shape
    eye = jnp.eye(dim, dtype=basis.dtype)
    chosen: list[jnp.ndarray] = []

    def orthogonalize(vec: jnp.ndarray) -> jnp.ndarray:
        for prev in chosen:
            vec = vec - prev * jnp.vdot(prev, vec)
        return vec

    for i in range(dim):
        e = eye[:, i]
        vec = basis @ (basis.T @ e)
        vec = orthogonalize(vec)
        norm = jnp.linalg.norm(vec)
        if float(norm) > eps:
            chosen.append(vec / norm)
            if len(chosen) == width:
                break

    if len(chosen) < width:
        for i in range(width):
            vec = orthogonalize(basis[:, i])
            norm = jnp.linalg.norm(vec)
            if float(norm) > eps:
                chosen.append(vec / norm)
                if len(chosen) == width:
                    break

    if len(chosen) != width:
        raise RuntimeError(f"Failed to canonicalize subspace of width {width}")

    return _canonicalize_column_signs(jnp.stack(chosen, axis=1), eps=eps)


def _canonicalize_eigenbasis(eigvals: jnp.ndarray, eigvecs: jnp.ndarray, group_rtol: float = 1e-5, eps: float = 1e-12) -> jnp.ndarray:
    order = jnp.flip(jnp.argsort(eigvals, stable=True))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    groups: list[tuple[int, int]] = []
    start = 0
    scale = max(float(jnp.max(jnp.abs(eigvals))) if eigvals.size > 0 else 0.0, 1.0)
    threshold = group_rtol * scale
    for idx in range(1, eigvals.shape[0] + 1):
        if idx == eigvals.shape[0] or abs(float(eigvals[idx - 1]) - float(eigvals[idx])) > threshold:
            groups.append((start, idx))
            start = idx

    canonical_groups = [_canonicalize_subspace(eigvecs[:, start:end], eps=eps) for start, end in groups]
    return jnp.concatenate(canonical_groups, axis=1) if canonical_groups else eigvecs


def init_preconditioner(grad: jnp.ndarray, hparams: SOAPHyperParams) -> dict:
    gg: list[jnp.ndarray | None] = []
    if grad.ndim == 1:
        if (not hparams.precondition_1d) or int(grad.shape[0]) > hparams.max_precond_dim:
            gg.append(_empty_like_torch_skip())
        else:
            gg.append(jnp.zeros((grad.shape[0], grad.shape[0]), dtype=grad.dtype))
    else:
        working_grad = merge_dims(grad, hparams.max_precond_dim, hparams.data_format) if hparams.merge_dims else grad
        for sh in working_grad.shape:
            if int(sh) > hparams.max_precond_dim:
                gg.append(_empty_like_torch_skip())
            else:
                gg.append(jnp.zeros((sh, sh), dtype=grad.dtype))

    return {
        "step": 0,
        "exp_avg": jnp.zeros_like(grad),
        "exp_avg_sq": jnp.zeros_like(grad),
        "GG": tuple(gg),
        "Q": None,
        "precondition_frequency": hparams.precondition_frequency,
        "shampoo_beta": hparams.shampoo_beta if hparams.shampoo_beta >= 0 else hparams.betas[1],
    }


def _project_impl(
    grad: jnp.ndarray,
    q_mats: tuple[jnp.ndarray | None, ...],
    hparams: SOAPHyperParams,
    *,
    back: bool,
) -> jnp.ndarray:
    original_shape = grad.shape
    permuted_shape = None
    if hparams.merge_dims:
        if hparams.data_format == "channels_last" and grad.ndim == 4:
            permuted_shape = jnp.transpose(grad, (0, 3, 1, 2)).shape
        grad = merge_dims(grad, hparams.max_precond_dim, hparams.data_format)

    for mat in q_mats:
        if _is_active_matrix(mat):
            axes = ([0], [1]) if back else ([0], [0])
            grad = jnp.tensordot(grad, mat, axes=axes)
        else:
            permute_order = tuple(list(range(1, grad.ndim)) + [0])
            grad = jnp.transpose(grad, permute_order)

    if hparams.merge_dims:
        if hparams.data_format == "channels_last" and len(original_shape) == 4:
            grad = jnp.transpose(grad.reshape(permuted_shape), (0, 2, 3, 1))
        else:
            grad = grad.reshape(original_shape)
    return grad


def project(grad: jnp.ndarray, entry: dict, hparams: SOAPHyperParams) -> jnp.ndarray:
    return _project_impl(grad, entry["Q"], hparams, back=False)


def project_back(grad: jnp.ndarray, entry: dict, hparams: SOAPHyperParams) -> jnp.ndarray:
    return _project_impl(grad, entry["Q"], hparams, back=True)


def get_orthogonal_matrix(gg: tuple[jnp.ndarray | None, ...]) -> tuple[jnp.ndarray | None, ...]:
    q_mats: list[jnp.ndarray | None] = []
    for matrix in gg:
        if not _is_active_matrix(matrix):
            q_mats.append(_empty_like_torch_skip())
            continue
        eigvals, q = jnp.linalg.eigh(matrix + 1e-30 * jnp.eye(matrix.shape[0], dtype=matrix.dtype))
        q_mats.append(_canonicalize_eigenbasis(eigvals, q))
    return tuple(q_mats)


def get_orthogonal_matrix_qr(entry: dict, hparams: SOAPHyperParams) -> tuple[tuple[jnp.ndarray | None, ...], jnp.ndarray]:
    exp_avg_sq = entry["exp_avg_sq"]
    orig_shape = exp_avg_sq.shape
    permuted_shape = None
    if hparams.merge_dims:
        if hparams.data_format == "channels_last" and len(orig_shape) == 4:
            permuted_shape = jnp.transpose(exp_avg_sq, (0, 3, 1, 2)).shape
        exp_avg_sq = merge_dims(exp_avg_sq, hparams.max_precond_dim, hparams.data_format)

    q_mats: list[jnp.ndarray | None] = []
    for idx, (matrix, orth) in enumerate(zip(entry["GG"], entry["Q"])):
        if not _is_active_matrix(matrix):
            q_mats.append(_empty_like_torch_skip())
            continue
        est_eig = jnp.diag(orth.T @ matrix @ orth)
        sort_idx = jnp.flip(jnp.argsort(est_eig, stable=True))
        exp_avg_sq = jnp.take(exp_avg_sq, sort_idx, axis=idx)
        est_eig = est_eig[sort_idx]
        orth = orth[:, sort_idx]
        power_iter = matrix @ orth
        q, _ = jnp.linalg.qr(power_iter)
        q_mats.append(_canonicalize_eigenbasis(est_eig, q))

    if hparams.merge_dims:
        if hparams.data_format == "channels_last" and len(orig_shape) == 4:
            exp_avg_sq = jnp.transpose(exp_avg_sq.reshape(permuted_shape), (0, 2, 3, 1))
        else:
            exp_avg_sq = exp_avg_sq.reshape(orig_shape)
    return tuple(q_mats), exp_avg_sq


def collect_matrix_arrays(prefix: str, mats: tuple[jnp.ndarray | None, ...] | None) -> tuple[dict[str, jnp.ndarray], list[int]]:
    arrays: dict[str, jnp.ndarray] = {}
    indices: list[int] = []
    if mats is None:
        return arrays, indices
    for idx, mat in enumerate(mats):
        if not _is_active_matrix(mat):
            continue
        arrays[f"{prefix}{idx}"] = mat
        indices.append(idx)
    return arrays, indices


def debug_parameter_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    entry: dict,
    hparams: SOAPHyperParams,
) -> tuple[dict[str, jnp.ndarray], dict]:
    arrays: dict[str, jnp.ndarray] = {
        "param_before": param,
        "grad": grad,
    }
    meta = {
        "step_before": int(entry.get("step", 0)),
        "had_q": bool(entry.get("Q") is not None),
        "precondition_frequency": int(entry.get("precondition_frequency", hparams.precondition_frequency)),
    }

    gg_prev, gg_prev_indices = collect_matrix_arrays("gg_prev_", entry.get("GG"))
    arrays.update(gg_prev)
    meta["gg_prev_indices"] = gg_prev_indices

    if entry.get("Q") is None:
        return arrays, meta

    q_prev, q_prev_indices = collect_matrix_arrays("q_prev_", entry["Q"])
    arrays.update(q_prev)
    meta["q_prev_indices"] = q_prev_indices

    grad_projected = project(grad, entry, hparams)
    beta1, beta2 = hparams.betas
    step_num = int(entry["step"]) + 1
    exp_avg_prev = entry["exp_avg"]
    exp_avg_sq_prev = entry["exp_avg_sq"]
    exp_avg_next = exp_avg_prev * beta1 + grad_projected * (1.0 - beta1)
    exp_avg_sq_next = exp_avg_sq_prev * beta2 + jnp.square(grad_projected) * (1.0 - beta2)
    denom = jnp.sqrt(exp_avg_sq_next) + hparams.eps

    step_size = hparams.lr
    if hparams.correct_bias:
        bias_correction1 = 1.0 - beta1 ** step_num
        bias_correction2 = 1.0 - beta2 ** step_num
        step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

    norm_grad = project_back(exp_avg_next / denom, entry, hparams)
    if hparams.normalize_grads:
        norm_grad = norm_grad / (1e-30 + jnp.sqrt(jnp.mean(jnp.square(norm_grad))))

    param_after_formula = param - step_size * norm_grad
    if hparams.weight_decay > 0.0:
        param_after_formula = param_after_formula + (-hparams.lr * hparams.weight_decay) * param_after_formula

    arrays.update({
        "grad_projected": grad_projected,
        "exp_avg_prev": exp_avg_prev,
        "exp_avg_sq_prev": exp_avg_sq_prev,
        "exp_avg_next": exp_avg_next,
        "exp_avg_sq_next": exp_avg_sq_next,
        "denom": denom,
        "norm_grad": norm_grad,
        "param_after_formula": param_after_formula,
    })
    meta["step_num"] = step_num
    meta["step_size"] = float(step_size)
    return arrays, meta


def debug_post_state(param: jnp.ndarray, entry: dict) -> tuple[dict[str, jnp.ndarray], dict]:
    arrays: dict[str, jnp.ndarray] = {
        "param_after_actual": param,
    }
    meta = {
        "step_after": int(entry.get("step", 0)),
        "had_q_after": bool(entry.get("Q") is not None),
    }
    if "exp_avg" in entry:
        arrays["exp_avg_post"] = entry["exp_avg"]
    if "exp_avg_sq" in entry:
        arrays["exp_avg_sq_post"] = entry["exp_avg_sq"]
    gg_post, gg_post_indices = collect_matrix_arrays("gg_post_", entry.get("GG"))
    arrays.update(gg_post)
    meta["gg_post_indices"] = gg_post_indices
    q_post, q_post_indices = collect_matrix_arrays("q_post_", entry.get("Q"))
    arrays.update(q_post)
    meta["q_post_indices"] = q_post_indices
    return arrays, meta


def update_preconditioner(grad: jnp.ndarray, entry: dict, hparams: SOAPHyperParams) -> dict:
    updated = dict(entry)
    if updated["Q"] is not None:
        updated["exp_avg"] = project_back(updated["exp_avg"], updated, hparams)

    gg = list(updated["GG"])
    if grad.ndim == 1:
        if hparams.precondition_1d and int(grad.shape[0]) <= hparams.max_precond_dim and _is_active_matrix(gg[0]):
            outer = jnp.outer(grad, grad)
            gg[0] = _lerp(gg[0], outer, 1.0 - updated["shampoo_beta"])
    else:
        working_grad = merge_dims(grad, hparams.max_precond_dim, hparams.data_format) if hparams.merge_dims else grad
        for idx, sh in enumerate(working_grad.shape):
            if int(sh) > hparams.max_precond_dim or not _is_active_matrix(gg[idx]):
                continue
            contract_dims = tuple(axis for axis in range(working_grad.ndim) if axis != idx)
            outer = jnp.tensordot(working_grad, working_grad, axes=(contract_dims, contract_dims))
            gg[idx] = _lerp(gg[idx], outer, 1.0 - updated["shampoo_beta"])

    updated["GG"] = tuple(gg)

    if updated["Q"] is None:
        updated["Q"] = get_orthogonal_matrix(updated["GG"])
    if updated["step"] > 0 and updated["step"] % updated["precondition_frequency"] == 0:
        updated["Q"], updated["exp_avg_sq"] = get_orthogonal_matrix_qr(updated, hparams)
    if updated["step"] > 0:
        updated["exp_avg"] = project(updated["exp_avg"], updated, hparams)

    return updated


def init_state(params: dict[str, jnp.ndarray], hparams: SOAPHyperParams) -> dict[str, dict]:
    return {name: init_preconditioner(param, hparams) for name, param in sorted(params.items())}


def step(
    params: dict[str, jnp.ndarray],
    grads: dict[str, jnp.ndarray],
    state: dict[str, dict],
    hparams: SOAPHyperParams,
) -> tuple[dict[str, jnp.ndarray], dict[str, dict]]:
    new_params: dict[str, jnp.ndarray] = {}
    new_state: dict[str, dict] = {}
    beta1, beta2 = hparams.betas

    for name, param in sorted(params.items()):
        grad = grads.get(name)
        if grad is None:
            new_params[name] = param
            new_state[name] = state.get(name, init_preconditioner(param, hparams))
            continue

        entry = dict(state.get(name, init_preconditioner(param, hparams)))
        if entry["Q"] is None:
            entry = update_preconditioner(grad, entry, hparams)
            new_params[name] = param
            new_state[name] = entry
            continue

        grad_projected = project(grad, entry, hparams)
        step_num = int(entry["step"]) + 1
        exp_avg = entry["exp_avg"] * beta1 + grad_projected * (1.0 - beta1)
        exp_avg_sq = entry["exp_avg_sq"] * beta2 + jnp.square(grad_projected) * (1.0 - beta2)
        denom = jnp.sqrt(exp_avg_sq) + hparams.eps

        step_size = hparams.lr
        if hparams.correct_bias:
            bias_correction1 = 1.0 - beta1 ** step_num
            bias_correction2 = 1.0 - beta2 ** step_num
            step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

        norm_grad = project_back(exp_avg / denom, entry, hparams)
        if hparams.normalize_grads:
            norm_grad = norm_grad / (1e-30 + jnp.sqrt(jnp.mean(jnp.square(norm_grad))))

        new_param = param - step_size * norm_grad
        if hparams.weight_decay > 0.0:
            new_param = new_param + (-hparams.lr * hparams.weight_decay) * new_param

        entry["step"] = step_num
        entry["exp_avg"] = exp_avg
        entry["exp_avg_sq"] = exp_avg_sq
        entry = update_preconditioner(grad, entry, hparams)

        new_params[name] = new_param
        new_state[name] = entry

    return new_params, new_state
