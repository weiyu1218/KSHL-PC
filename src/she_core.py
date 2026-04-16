import numpy as np
import scipy.sparse as sp
import time
from sklearn.preprocessing import normalize

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sparse_dot_mkl import dot_product_mkl
    HAS_MKL = True
except ImportError:
    HAS_MKL = False

from scipy.fft import fft, ifft


class SHEConfig:
    def __init__(self, embedding_dim=32, svd_rank=32, svd_tol=0,
                 window=10, alpha=0.1, sketch_dim=128,
                 poly_deg=3, fit_sample=10, seed=None, vol_type='traditional'):
        self.embedding_dim = embedding_dim
        self.svd_rank = svd_rank
        self.svd_tol = svd_tol
        self.window = window
        self.alpha = alpha
        self.sketch_dim = sketch_dim
        self.poly_deg = poly_deg
        self.fit_sample = fit_sample
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.vol_type = vol_type  # 'traditional': sum of d(v); 'normalized': sum of D_e^{-1/2}HD_v^{-1/2}


def count_sketch_matrix(M, s, config):
    n, d = M.shape
    hash_indices = config.rng.integers(0, s, d)
    hash_signs = config.rng.choice([-1, 1], d)
    S = sp.csr_matrix((hash_signs, hash_indices, np.arange(d+1)), shape=(d, s))
    return S


def fit_coeffs(U, func, config):
    sample_ids = config.rng.choice(U.shape[0], config.fit_sample, replace=False)
    ps = np.full(config.fit_sample, U.shape[0]/config.fit_sample)
    U_sample = U[sample_ids]
    X = (U_sample @ U.T).flatten()
    X_vander = np.vander(X, config.poly_deg + 1, increasing=True)
    ps = np.concatenate([np.full(U.shape[0], p) for p in ps])
    Xt_D = X_vander.T * ps
    w = np.zeros(config.poly_deg + 1)
    U_norms = np.square(U).sum(axis=1)
    for i in range(1, config.poly_deg + 1):
        w[i] = np.sqrt(config.poly_deg*(2+3**i)*((U_norms**i).sum()**2)/config.sketch_dim)
    func = np.vectorize(func, otypes=[np.float64])
    Y = func(X)
    coeffs = np.linalg.inv(Xt_D @ X_vander + np.diag(w ** 2)) @ Xt_D @ Y
    return coeffs


def tensor_sketches(U, coeffs, sketch_dim, config):
    n, d = U.shape
    tu = [np.ones((n, 1), np.float64)]
    S = count_sketch_matrix(U, sketch_dim, config)
    U_sketch = U @ S
    tu.append(U_sketch)
    fu = fft(tu[-1])
    fu0 = fu
    c_vecs = [[coeffs[0]]]
    c_vecs.append(np.full(sketch_dim, coeffs[1]))
    for i in range(2, len(coeffs)):
        fu = fu0 * fu
        tu.append(ifft(fu).real)
        c_vecs.append(np.full(sketch_dim, coeffs[i]))
    c_diag = sp.diags(np.concatenate(c_vecs))
    tu = np.concatenate([_ for _ in tu], axis=1)
    return tu, c_diag


def pts_tlog(U, config):
    tlog_func = lambda x: np.log(x) if x > 1 else 0
    coeffs = fit_coeffs(U, func=tlog_func, config=config)
    return tensor_sketches(U, coeffs=coeffs, sketch_dim=config.sketch_dim, config=config)


def she_embed(H0, X, config):
    """
    SHE embedding using structural hyperedges.

    Args:
        H0: Structural hyperedge incidence matrix (m, n) where m=num_hyperedges, n=num_vertices
        X: Attribute features (for API compatibility)
        config: SHEConfig object

    Returns:
        Z_v: Vertex embeddings (n, embedding_dim)
        elapsed_time: Computation time in seconds
    """
    start_time = time.perf_counter()

    H = H0
    m, n = H.shape

    # Compute vertex degrees
    deg_vec = H.T.dot(np.ones(m))
    deg_vec[deg_vec == 0] = 1
    deg_alpha = deg_vec ** (-0.5)
    D_v = sp.diags(deg_alpha, format='csr')

    # Compute hyperedge degrees
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge == 0] = 1
    D_e = sp.diags(deg_edge ** (-0.5))

    # Normalized hypergraph incidence
    norm_H = D_e @ H @ D_v

    # Volume: scale factor for structural basis F
    # 'traditional': vol(V) = sum_v d(v) = H.sum() (classical spectral hypergraph theory)
    # 'normalized':  sum of D_e^{-1/2}HD_v^{-1/2} (paper Eq.7 stated definition)
    if getattr(config, 'vol_type', 'traditional') == 'normalized':
        vol = float(norm_H.sum())
    else:
        vol = float(H.sum())

    if HAS_MKL:
        def linear_operator(x):
            return dot_product_mkl(norm_H, x, cast=True)
        def linear_operator1(x):
            return dot_product_mkl(norm_H.T, x, cast=True)
        H_LO = sp.linalg.LinearOperator((m, n), matvec=linear_operator, rmatvec=linear_operator1)
    else:
        H_LO = norm_H

    U, s, VT = sp.linalg.svds(H_LO, min(config.svd_rank, n//2), tol=config.svd_tol, random_state=config.rng)
    sigma_sq = s ** 2
    sigma_T = np.full_like(sigma_sq, config.alpha)
    for i in range(1, config.window + 1):
        sigma_T += config.alpha * (1 - config.alpha) ** i * sigma_sq ** i

    F_v = np.sqrt(vol) * sp.diags(np.sqrt(sigma_T)).dot(VT @ D_v).T
    Y_v, Th_v = pts_tlog(F_v, config)
    YTh_v = Y_v @ Th_v

    def laplacian_operator(v):
        return Y_v @ (YTh_v.T @ v)

    Gamma_v_LO = sp.linalg.LinearOperator((n, n), matvec=laplacian_operator)
    _, Z_v = sp.linalg.eigsh(Gamma_v_LO, min(config.embedding_dim, n//2), tol=config.svd_tol, which='LM')
    Z_v = normalize(Z_v, norm='l2', axis=1)

    elapsed_time = time.perf_counter() - start_time

    return Z_v, elapsed_time



