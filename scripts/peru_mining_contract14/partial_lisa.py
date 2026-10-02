"""Contract 12: joint active-support randomization, frozen monthly gamma.

Unlike GPUESDA's focal-fixed conditional engine, every active value (including
the focal) is permuted. Geography, Empty values and nuisance adjustment are fixed.
NumPy PCG64 generates one permutation per draw; CuPy evaluates the full fixed W.
Draw generation is independent of arithmetic batch size.
"""
import hashlib
import time
import numpy as np

R_PERM = 19999
P_FLOOR = 1.0 / (R_PERM + 1)


def fingerprint(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def w_identity(w):
    return [fingerprint(v) for v in (w.data, w.indices, w.indptr)]


def pseudo_p(upper, R=R_PERM):
    # GPUESDA convention: upper is >= observed; lower is its strict complement.
    upper = np.asarray(upper)
    assert np.all((upper >= 0) & (upper <= R))
    return (np.minimum(upper, R-upper)+1.0)/(R+1.0)


def bh(p):
    p = np.asarray(p, dtype=float)
    assert np.isfinite(p).all() and np.all((p > 0) & (p <= 1))
    order = np.argsort(p, kind='stable')
    # Same operation ordering as statsmodels for exact float64 agreement.
    adjusted = p[order] / (np.arange(1, len(p)+1)/float(len(p)))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1)
    q = np.empty_like(p)
    q[order] = adjusted
    return q


def draw_orders(m, R, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(m) for _ in range(R)]).astype(np.int32)


def validate_draw(z, active, permuted):
    assert np.array_equal(permuted[:, ~active], np.broadcast_to(z[~active], permuted[:, ~active].shape)), 'Empty moved'
    assert np.array_equal(np.sort(permuted[:, active], axis=1), np.broadcast_to(np.sort(z[active]), permuted[:, active].shape)), 'Donor multiset changed'


def cpu_reference(z, W, active, N, gamma, orders):
    """Independent draw-by-draw full-geography NumPy/SciPy calculation."""
    out = []
    for order in orders:
        perm = z.copy()
        perm[active] = z[active][order]
        x = perm[active]-np.mean(perm[active])
        adjusted = (W @ perm)[active]-gamma*N[active]
        y = adjusted-np.mean(adjusted)
        out.append((active.sum()-1)*x*y/np.dot(x, x))
    return np.asarray(out)


def simulate(z, W, active, N, gamma, observed, *, seed, R=R_PERM,
             batch_size=256, keep=False, audit=False, orders=None):
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    assert cp.cuda.runtime.getDeviceCount() > 0
    assert z.dtype == np.float64 and active.dtype == bool
    assert len(z) == W.shape[0] == W.shape[1] and active.sum() > 1
    assert np.isfinite(z).all() and W.format == 'csr'
    before = [fingerprint(z), fingerprint(active), fingerprint(N), w_identity(W)]
    started = time.perf_counter()
    if orders is None:
        orders = draw_orders(int(active.sum()), R, seed)
    assert orders.shape == (R, active.sum())
    assert np.array_equal(np.sort(orders, axis=1), np.broadcast_to(np.arange(active.sum()), orders.shape))
    ids = cp.asarray(np.flatnonzero(active))
    zg, ng, wg = cp.asarray(z), cp.asarray(N[active]), csr_matrix(W)
    denom = float(np.sum((z[active]-z[active].mean())**2))
    assert denom > 0
    upper = np.zeros(active.sum(), dtype=np.int64)
    saved, peak, cpu_error = [], 0, 0.0
    for start in range(0, R, batch_size):
        rows = orders[start:start+batch_size]
        perm = cp.broadcast_to(zg, (len(rows), len(z))).copy()
        perm[:, ids] = zg[ids][cp.asarray(rows)]
        lag = (wg @ perm.T).T[:, ids]
        x = perm[:, ids]-cp.mean(perm[:, ids], axis=1, keepdims=True)
        adjusted = lag-gamma*ng
        y = adjusted-cp.mean(adjusted, axis=1, keepdims=True)
        sim = (active.sum()-1)*x*y/denom
        host = cp.asnumpy(sim)
        assert np.isfinite(host).all()
        upper += np.count_nonzero(host >= observed[None, :], axis=0)
        if audit:
            validate_draw(z, active, cp.asnumpy(perm))
            ref = cpu_reference(z, W, active, N, gamma, rows)
            np.testing.assert_allclose(host, ref, atol=1e-12, rtol=1e-12)
            cpu_error = max(cpu_error, float(np.max(np.abs(host-ref))))
        if keep:
            saved.append(host)
        peak = max(peak, cp.get_default_memory_pool().total_bytes())
    cp.cuda.Stream.null.synchronize()
    assert before == [fingerprint(z), fingerprint(active), fingerprint(N), w_identity(W)]
    return {'upper':upper, 'p':pseudo_p(upper, R),
            'simulations':np.concatenate(saved) if keep else None,
            'audit':{'R':R,'seed':seed,'batch_size':batch_size,'seconds':time.perf_counter()-started,
                     'sampled_peak_cupy_pool_bytes':peak,'max_cpu_error':cpu_error,
                     'draw_sha256':fingerprint(orders),'W_and_masks_unchanged':True}}
