import torch
import torch.nn.functional as F
import math

from nika import load_video_frames

def _unfold(x: torch.Tensor, mode: int) -> torch.Tensor:
    x = x.movedim(mode, 0)
    return x.reshape(x.shape[0], -1)


@torch.no_grad()
def _top_svals(A: torch.Tensor, k: int, niter: int = 2) -> torch.Tensor:
    # Keep wrapper for small/straightforward cases; prefer svd_lowrank when
    # the entire unfolding fits comfortably in memory.
    m, n = A.shape
    q = min(k, m, n)
    if q <= 0:
        return torch.empty((0,), device=A.device, dtype=torch.float32)

    A32 = A.to(torch.float32)
    _, S, _ = torch.svd_lowrank(A32, q=q, niter=niter)
    return S  # float32


def _top_svals_via_gram(A: torch.Tensor, k: int, chunk_cols: int = 65536) -> torch.Tensor:
    """
    Compute top singular values for unfolding A (shape [d, M]) by forming
    the Gram matrix G = A @ A.T in column chunks. This avoids creating
    large intermediate matrices when M is huge (spatial dims).
    Returns descending singular values (float32).
    """
    d, M = A.shape
    if M == 0:
        return torch.empty((0,), device=A.device, dtype=torch.float32)

    # accumulate in float64 for numerical stability
    G = torch.zeros((d, d), device=A.device, dtype=torch.float64)
    for s in range(0, M, chunk_cols):
        e = min(s + chunk_cols, M)
        # A[:, s:e] is a view; convert to float32 to reduce memory when possible
        chunk = A[:, s:e].to(torch.float32)
        # chunk @ chunk.T -> (d, d)
        G += (chunk @ chunk.T).to(torch.float64)

    # eigendecompose Gram (small: d x d)
    vals, _ = torch.linalg.eigh(G)
    vals = vals.clamp_min(0.0)
    svals = torch.sqrt(vals).to(torch.float32)
    # eigenvalues are ascending; reverse to descending singular values
    svals = svals.flip(0)
    q = min(k, svals.numel())
    return svals[:q]


def _top_svecs_via_gram(A: torch.Tensor, k: int, chunk_cols: int = 65536):
    """
    Return top-k left singular vectors and singular values for A (d, M) using Gram.
    Returns (U, svals) where U is (d, k) (float32) and svals length k.
    """
    d, M = A.shape
    if M == 0 or k <= 0:
        return torch.empty((d, 0), device=A.device, dtype=torch.float32), torch.empty((0,), device=A.device)

    G = torch.zeros((d, d), device=A.device, dtype=torch.float64)
    for s in range(0, M, chunk_cols):
        e = min(s + chunk_cols, M)
        chunk = A[:, s:e].to(torch.float32)
        G += (chunk @ chunk.T).to(torch.float64)

    vals, vecs = torch.linalg.eigh(G)
    vals = vals.clamp_min(0.0)
    # reverse order to descending
    vals = vals.flip(0)
    vecs = vecs.flip(1)
    q = min(k, vals.numel())
    svals = torch.sqrt(vals[:q]).to(torch.float32)
    U = vecs[:, :q].to(torch.float32)
    return U, svals


def _top_svals_via_gram_from_tensor(x: torch.Tensor, mode: int, k: int, chunk_cols: int = 65536):
    """
    Compute top singular values for mode-unfolding of 4D tensor `x` without
    constructing the full unfolding. Tiles the other dims to accumulate the
    Gram matrix G = A @ A.T in chunks.
    """
    assert x.ndim == 4
    dims = list(x.shape)
    d = dims[mode]
    # list other axes in order
    other = [i for i in range(4) if i != mode]
    a, b, c = [dims[i] for i in other]

    # permute so mode is first: x_perm shape (d, a, b, c)
    x_perm = x.permute(mode, *other)

    # accumulate Gram in float64
    G = torch.zeros((d, d), device=x.device, dtype=torch.float64)

    # choose tile sizes for b and c so that a * tile_b * tile_c <= chunk_cols
    max_area = max(1, chunk_cols // max(1, a))
    tile_b = min(b, max(1, int(math.sqrt(max_area))))
    tile_c = max(1, max_area // tile_b)

    for b0 in range(0, b, tile_b):
        b1 = min(b, b0 + tile_b)
        for c0 in range(0, c, tile_c):
            c1 = min(c, c0 + tile_c)
            # chunk shape: (d, a, tile_b, tile_c)
            chunk = x_perm[:, :, b0:b1, c0:c1]
            # reshape to (d, a * tile_b * tile_c)
            chunk_mat = chunk.reshape(d, -1).to(torch.float32)
            G += (chunk_mat @ chunk_mat.T).to(torch.float64)

    vals, _ = torch.linalg.eigh(G)
    vals = vals.clamp_min(0.0)
    svals = torch.sqrt(vals).to(torch.float32)
    svals = svals.flip(0)
    q = min(k, svals.numel())
    return svals[:q]


def _unfold_3d(x: torch.Tensor, mode: int) -> torch.Tensor:
    x = x.movedim(mode, 0)
    return x.reshape(x.shape[0], -1)


def _top_svals_via_gram_from_tensor_3d(x: torch.Tensor, mode: int, k: int, chunk_cols: int = 65536):
    """
    Compute top singular values for mode-unfolding of 3D tensor `x` without
    constructing the full unfolding. Tiles the other dims to accumulate the
    Gram matrix G = A @ A.T in chunks.
    """
    assert x.ndim == 3
    dims = list(x.shape)
    d = dims[mode]
    # list other axes in order
    other = [i for i in range(3) if i != mode]
    a, b = [dims[i] for i in other]

    # permute so mode is first: x_perm shape (d, a, b)
    x_perm = x.permute(mode, *other)

    # accumulate Gram in float64
    G = torch.zeros((d, d), device=x.device, dtype=torch.float64)

    # tile b so that a * tile_b <= chunk_cols
    tile_b = min(b, max(1, chunk_cols // max(1, a)))

    for b0 in range(0, b, tile_b):
        b1 = min(b, b0 + tile_b)
        # chunk shape: (d, a, tile_b)
        chunk = x_perm[:, :, b0:b1]
        # reshape to (d, a * tile_b)
        chunk_mat = chunk.reshape(d, -1).to(torch.float32)
        G += (chunk_mat @ chunk_mat.T).to(torch.float64)

    vals, _ = torch.linalg.eigh(G)
    vals = vals.clamp_min(0.0)
    svals = torch.sqrt(vals).to(torch.float32)
    svals = svals.flip(0)
    q = min(k, svals.numel())
    return svals[:q]


@torch.no_grad()
def rank_estimate_3d(
    feature_tensor: torch.Tensor,               # (C,H,W) float
    energy: float = 0.98,
    kmax: tuple[int, int, int] = (256, 256, 256),
    rmax: tuple[int | None, int | None, int | None] = (512, 512, 512),
):
    assert feature_tensor.ndim == 3, "expected (C,H,W)"
    C, H, W = feature_tensor.shape
    dev = feature_tensor.device

    x = feature_tensor.to(torch.float32)
    x = x - x.mean()

    x = x.cpu()

    ranks = []
    svals = []

    energy_threshold = energy
    for mode in range(3):
        # use tiled Gram accumulation directly from the 3D tensor
        S = _top_svals_via_gram_from_tensor_3d(x, mode, k=kmax[mode])
        r = _rank_for_energy(S, energy=energy_threshold, rmin=1, rmax=rmax[mode])
        ranks.append(r)
        svals.append(S.detach().cpu())

    return tuple(ranks), svals


def tune_energy_for_center_3d(
    feature_tensor: torch.Tensor,
    center_target: int = 750000,
    energy_lo: float = 0,
    energy_hi: float = 1,
    tol: float = 0.05,
    max_iters: int = 20,
    **rank_kwargs,
):
    """
    Binary-search energy threshold so that product(ranks) approximates center_target.
    Returns (energy, ranks, product, svals).
    """
    lo = energy_lo
    hi = energy_hi
    best = None
    for it in range(max_iters):
        mid = (lo + hi) / 2.0
        ranks, svals = rank_estimate_3d(feature_tensor, energy=mid, **rank_kwargs)
        prod = 1
        for r in ranks:
            prod *= int(r)
        # record best
        if best is None or abs(prod - center_target) < abs(best[2] - center_target):
            best = (mid, ranks, prod, svals)

        # check tolerance
        rel_err = abs(prod - center_target) / float(center_target)
        print(f"iter {it}: energy={mid:.6f}, ranks={ranks}, prod={prod}, rel_err={rel_err:.4f}")
        if rel_err <= tol:
            break

        # monotonic: increasing energy -> larger ranks -> larger product
        if prod > center_target:
            # product too large -> reduce energy
            hi = mid
        else:
            lo = mid

    return best


def estimate_feature_grid_rank(feature_grid: torch.Tensor, k: int = 128) -> torch.Tensor:
    """
    Estimate the rank of a feature grid (C, H, W) by computing top singular values
    of the unfolded matrix (C, H*W).
    """
    assert feature_grid.ndim == 3, "expected (C, H, W)"
    A = feature_grid.reshape(feature_grid.shape[0], -1)  # (C, H*W)
    return _top_svals_via_gram(A, k)


def rank_estimate_grid4d(
    grid_tensor: torch.Tensor,               # (T,C,H,W) float
    energy: float = 0.98,
    kmax: tuple[int, int, int, int] = (128, 128, 256, 256),
    rmax: tuple[int | None, int | None, int | None, int | None] = (128, None, 512, 512),
):
    """
    Estimate Tucker ranks for a feature grid shaped (T, C, H, W). This treats
    `C` as the feature dimension and includes the temporal axis `T` in the
    per-mode rank estimation.
    Returns (ranks, svals) where svals is a list of singular-value tensors per mode.
    """
    assert grid_tensor.ndim == 4, "expected (T, C, H, W)"
    T, C, H, W = grid_tensor.shape

    x = grid_tensor.to(torch.float32)
    if x.max() > 2.0:
        x = x / 255.0
    # normalize per-channel (C) across (T,H,W) for stability
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std = x.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std

    x = x.cpu()

    ranks = []
    svals = []

    for mode in range(4):
        S = _top_svals_via_gram_from_tensor(x, mode, k=kmax[mode])
        r = _rank_for_energy(S, energy=energy, rmin=1, rmax=rmax[mode])
        ranks.append(r)
        svals.append(S.detach().cpu())

    return tuple(ranks), svals


def tune_energy_for_center_grid4d(
    grid_tensor: torch.Tensor,
    center_target: int = 750000,
    energy_lo: float = 0.0,
    energy_hi: float = 1.0,
    tol: float = 0.05,
    max_iters: int = 20,
    **rank_kwargs,
):
    """
    Binary-search energy threshold so that product(ranks) approximates center_target
    for a (T,C,H,W) feature grid. Returns (energy, ranks, product, svals).
    """
    lo = energy_lo
    hi = energy_hi
    best = None
    for it in range(max_iters):
        mid = (lo + hi) / 2.0
        ranks, svals = rank_estimate_grid4d(grid_tensor, energy=mid, **rank_kwargs)
        prod = 1
        for r in ranks:
            prod *= int(r)
        if best is None or abs(prod - center_target) < abs(best[2] - center_target):
            best = (mid, ranks, prod, svals)
        rel_err = abs(prod - center_target) / float(center_target)
        print(f"iter {it}: energy={mid:.6f}, ranks={ranks}, prod={prod}, rel_err={rel_err:.4f}")
        if rel_err <= tol:
            break
        if prod > center_target:
            hi = mid
        else:
            lo = mid

    return best


def deflate_tensor(video_tensor: torch.Tensor, remove_ranks: tuple[int, int, int, int], chunk_cols: int = 65536, compute_device: str = 'cpu') -> torch.Tensor:
    """
    Project out the top `remove_ranks[mode]` left singular vectors from each mode
    (separately) and return the residual tensor. Works on CPU by default.
    """
    assert video_tensor.ndim == 4
    T, C, H, W = video_tensor.shape
    x = video_tensor.to(torch.float32)
    if x.max() > 2.0:
        x = x / 255.0
    x = x - x.mean()

    dev = torch.device(compute_device or 'cpu')
    x = x.to(dev)

    for mode in range(4):
        r = int(remove_ranks[mode])
        if r <= 0:
            continue
        A = _unfold(x, mode)  # (d, M)
        U, _ = _top_svecs_via_gram(A, k=r, chunk_cols=chunk_cols)
        if U.numel() == 0:
            continue
        # project: A_proj = U @ (U.T @ A)
        coef = U.T @ A  # (r, M)
        A_proj = U @ coef  # (d, M)
        A = A - A_proj
        x = _fold_from_unfold(A, (T, C, H, W), mode).to(dev)

    return x


def deflate_and_reestimate(video_tensor: torch.Tensor, remove_ranks: tuple[int, int, int, int], **rank_kwargs):
    """
    Remove top `remove_ranks` components and re-run rank_estimate on the residual.
    Returns (orig_ranks, residual_ranks, orig_svals, residual_svals).
    """
    # compute original ranks/svals
    orig_ranks, orig_svals = rank_estimate(video_tensor, **rank_kwargs)
    # deflate
    residual = deflate_tensor(video_tensor, remove_ranks, compute_device=rank_kwargs.get('compute_device', 'cpu'))
    # recompute ranks on residual
    res_ranks, res_svals = rank_estimate(residual, **rank_kwargs)
    return orig_ranks, res_ranks, orig_svals, res_svals


def _rank_for_energy(s: torch.Tensor, energy: float, rmin: int = 1, rmax: int | None = None) -> int:
    if s.numel() == 0:
        return rmin
    if rmax is None:
        rmax = s.numel()
    rmax = min(rmax, s.numel())

    ss = (s[:rmax] ** 2)
    denom = ss.sum().clamp_min(1e-12)
    cume = torch.cumsum(ss, dim=0) / denom
    mask = (cume >= energy)
    if mask.any():
        r = int(mask.nonzero(as_tuple=False)[0].item() + 1)
    else:
        # energy target not reached within rmax (e.g. energy==1.0); pick rmax
        r = int(rmax)
    return max(rmin, min(r, rmax))


@torch.no_grad()
def rank_estimate(
    video_tensor: torch.Tensor,               # (T,C,H,W) uint8 or float
    frame_sample: int = 32,
    energy: float = 0.98,
    kmax: tuple[int, int, int, int] = (128, 3, 256, 256),
    rmax: tuple[int | None, int | None, int | None, int | None] = (128, None, 512, 512),
):
    assert video_tensor.ndim == 4, "expected (T,C,H,W)"
    T, C, H, W = video_tensor.shape
    dev = video_tensor.device

    x = video_tensor.to(torch.float32)

    if x.max() > 2.0:
        x = x / 255.0
    x = x - x.mean()

    if frame_sample is not None and frame_sample < T:
        n = min(frame_sample, T)
        idx = torch.linspace(0, T - 1, steps=n, device=dev).round().long()
        x_sampled = x.index_select(0, idx)  # (n,C,H,W)
    else:
        x_sampled = x

    x_ds = x_sampled.cpu()

    ranks = []
    svals = []

    center_target = 750000

    energy_threshold = energy
    for mode in range(4):
        # use tiled Gram accumulation directly from the 4D tensor to avoid
        # creating a huge unfolded matrix in memory
        S = _top_svals_via_gram_from_tensor(x_ds, mode, k=kmax[mode])
        r = _rank_for_energy(S, energy=energy_threshold, rmin=1, rmax=rmax[mode])
        ranks.append(r)
        svals.append(S.detach().cpu())

    return tuple(ranks), svals


def tune_energy_for_center(
    video_tensor: torch.Tensor,
    center_target: int = 750000,
    energy_lo: float = 0,
    energy_hi: float = 1,
    tol: float = 0.05,
    max_iters: int = 20,
    **rank_kwargs,
):
    """
    Binary-search energy threshold so that product(ranks) approximates center_target.
    Returns (energy, ranks, product, svals).
    """
    lo = energy_lo
    hi = energy_hi
    best = None
    for it in range(max_iters):
        mid = (lo + hi) / 2.0
        ranks, svals = rank_estimate(video_tensor, energy=mid, **rank_kwargs)
        prod = 1
        for r in ranks:
            prod *= int(r)
        # record best
        if best is None or abs(prod - center_target) < abs(best[2] - center_target):
            best = (mid, ranks, prod, svals)

        # check tolerance
        rel_err = abs(prod - center_target) / float(center_target)
        print(f"iter {it}: energy={mid:.6f}, ranks={ranks}, prod={prod}, rel_err={rel_err:.4f}")
        if rel_err <= tol:
            break

        # monotonic: increasing energy -> larger ranks -> larger product
        if prod > center_target:
            # product too large -> reduce energy
            hi = mid
        else:
            lo = mid

    return best


def run_rank_analysis(video_tensor: torch.Tensor, center_target: int = 750000, compute_device: str = 'cpu'):
    """
    Run tuning on several representations independently and return a dict of results.
    Representations: 'real', 'fft_real', 'fft_imag', 'fft_mag', 'fft_complex'
    """
    results = {}
    vid = video_tensor.cpu()

    def _log_repr_stats(name: str, x: torch.Tensor):
        print(f"[{name}] shape={tuple(x.shape)}, min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std(unbiased=False):.6f}")

    def _log_svals(name: str, svals_list):
        print(f"SVals for {name}:")
        for mi, s in enumerate(svals_list):
            s = s.detach().cpu()
            topk = min(10, s.numel())
            vals = s[:topk].numpy().tolist()
            print(f"  mode {mi}: len={s.numel()}, top{topk}={vals}, energy_sum={float((s**2).sum()):.6e}")

    # Keep only the core analyses: real, fft_complex (Tucker on complex concat), feature_grid
    print("\n-- Analysis: real-domain --")
    _log_repr_stats('real', vid)
    # normalize per-channel for stability (video already scaled in main)
    def _zscore_per_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(eps)
        return (x - mean) / std

    real_norm = _zscore_per_channel(vid)
    best_real = tune_energy_for_center(
        real_norm,
        center_target=1000000,
        max_iters=20,
        frame_sample=None,
        kmax=(min(real_norm.shape[0], 300), real_norm.shape[1], real_norm.shape[2], real_norm.shape[3]),
        rmax=(128, None, 512, 512),
    )
    results['real'] = best_real
    _log_svals('real', best_real[3])

    # FFT complex: concat real+imag and run the same tucker-style rank estimation
    freq = torch.fft.fft2(vid.to(torch.float32), norm='ortho')
    freq_real = freq.real
    freq_imag = freq.imag
    freq_complex = torch.cat([freq_real, freq_imag], dim=1)
    print("\n-- Analysis: FFT complex (real+imag concat) --")
    _log_repr_stats('fft_complex', freq_complex)
    fc_norm = _zscore_per_channel(freq_complex)
    best_fft_complex = tune_energy_for_center(
        fc_norm,
        center_target=500000,
        max_iters=20,
        frame_sample=None,
        kmax=(min(fc_norm.shape[0], 300), fc_norm.shape[1], fc_norm.shape[2], fc_norm.shape[3]),
        rmax=(128, None, 512, 512),
    )
    results['fft_complex'] = best_fft_complex
    _log_svals('fft_complex', best_fft_complex[3])

    # Feature grid analysis on full (T, C, H, W) to include temporal rank
    print("\n-- Analysis: feature grid (T,C,H,W) --")
    _log_repr_stats('feature_grid', vid)
    best_fg = tune_energy_for_center_grid4d(
        vid,
        center_target=1000000,
        energy_lo=0.0,
        energy_hi=0.9999,
        max_iters=20,
        kmax=(min(vid.shape[0], 300), vid.shape[1], vid.shape[2], vid.shape[3]),
        rmax=(128, None, 512, 512),
    )
    results['feature_grid'] = best_fg
    _log_svals('feature_grid', best_fg[3])

    return results


if __name__ == "__main__":
    # Force loading and computation on CPU to avoid GPU OOMs
    device = "cuda:0"
    name = "yacht"
    vid = load_video_frames(f"static/benchmarks/uvg/{name}", device, max_frames=600, dtype=torch.uint8, normalize=False)
    vid = vid.cpu()[..., ::4, ::4].to(torch.float32) / 255.0  # downsample for faster analysis
    # Run the multi-representation analysis
    results = run_rank_analysis(vid, center_target=1400000, compute_device='cpu')
    print("\n=== Final Results ===")
    for rep, (energy, ranks, prod, svals) in results.items():
        print(f"{rep}: energy={energy:.6f}, ranks={ranks}, product={prod}")
