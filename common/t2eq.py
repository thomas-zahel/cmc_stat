
# t2eq.py
# Python implementation of the T2EQ test (Hotelling's T^2-based equivalence test)
# as described by Hoffelder and colleagues for dissolution profile similarity.
#
# References informing this implementation:
# - CRAN package T2EQ (function T2EQ.dissolution.profiles.hoffelder), which defines
#   the exact computation using a non-central F distribution with
#   df1=p, df2=N-1-p and ncp = (m*n/N) * (D' S^{-1} D), where D=(10,...,10).
#   See: https://github.com/cran/T2EQ/blob/master/R/T2EQ.dissolution.profiles.hoffelder.R
# - Hoffelder (2019), "Equivalence analyses of dissolution profiles with the Mahalanobis distance", Biometrical Journal.
# - The AAPS Journal (2022) meeting report on dissolution profile similarity analyses.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Try SciPy for noncentral F CDF/PPF. Fallback to mpmath-based implementation if SciPy is unavailable.
try:
    from scipy.stats import ncf
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

try:
    import mpmath as mp  # type: ignore
    _HAVE_MPMATH = True
except Exception:  # pragma: no cover
    _HAVE_MPMATH = False


@dataclass
class T2EQResult:
    p_value: float
    decision: bool
    alpha: float
    T2: float
    MD: float
    test_stat: float
    crit_value: float
    ncp: float
    df1: int
    df2: int
    m: int
    n: int
    p: int
    N: int
    mean_ref: np.ndarray
    mean_test: np.ndarray
    S_ref: np.ndarray
    S_test: np.ndarray
    S_pooled: np.ndarray
    eq_margin: float
    D: np.ndarray
    used_pinv: bool


class NumericalWarning(UserWarning):
    pass


def _cov_ddof1(x: np.ndarray) -> np.ndarray:
    """Sample covariance matrix with ddof=1 (to match R's var()) for rows=observations."""
    return np.cov(x, rowvar=False, ddof=1)


def _safe_inv(mat: np.ndarray, rcond: float = 1e-10) -> Tuple[np.ndarray, bool]:
    """Return inverse if well-conditioned, otherwise Moore-Penrose pseudoinverse.
    Returns (inv, used_pinv_flag).
    """
    try:
        inv = np.linalg.inv(mat)
        used_pinv = False
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(mat, rcond=rcond)
        used_pinv = True
    return inv, used_pinv


def _ncf_cdf_ppf_with_mpmath(x: Optional[float], q: Optional[float], df1: int, df2: int, ncp: float,
                              tol: float = 1e-10, max_terms: int = 2000) -> Tuple[Optional[float], Optional[float]]:
    """Compute CDF (if x is not None) and inverse CDF/quantile (if q is not None) of the non-central F distribution
    using the Poisson-weighted mixture representation in terms of central F distributions.

    This is used as a fallback when SciPy is not available. Accuracy is controlled by tol and max_terms.
    """
    if not _HAVE_MPMATH:
        raise RuntimeError("mpmath is required when SciPy is not available.")

    lam = ncp / 2.0  # Poisson mean for mixture weights

    def central_f_cdf(xx: float, a: int, b: int) -> float:
        # CDF of central F via regularized incomplete beta: I_{ (a*xx)/(a*xx + b) } (a/2, b/2)
        z = (a * xx) / (a * xx + b)
        return float(mp.betainc(a/2.0, b/2.0, 0, z, regularized=True))

    def ncf_cdf(xx: float) -> float:
        # Sum weights until tail mass < tol
        total = 0.0
        weight = mp.e**(-lam)
        k = 0
        # running sum of Poisson; use recursive formula for weights
        while k < max_terms:
            cdf_k = central_f_cdf(xx, df1 + 2 * k, df2)
            total += float(weight) * cdf_k
            # next weight
            k += 1
            weight *= lam / k
            if float(weight) < tol:
                break
        return min(max(total, 0.0), 1.0)

    def ncf_ppf(prob: float) -> float:
        # bracket search on x and bisection
        if prob <= 0.0:
            return 0.0
        if prob >= 1.0:
            return mp.inf
        lo, hi = 0.0, 1.0
        # expand hi until cdf(hi) >= prob
        while ncf_cdf(hi) < prob:
            hi *= 2.0
            if hi > 1e10:
                break
        # bisection
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            cmid = ncf_cdf(mid)
            if abs(cmid - prob) < 1e-8:
                return float(mid)
            if cmid < prob:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    cdf_val = ncf_cdf(x) if x is not None else None
    ppf_val = ncf_ppf(q) if q is not None else None
    return cdf_val, ppf_val


def t2eq_test(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05,
              D: Optional[np.ndarray] = None,
              use_pinv_if_needed: bool = True,
              rcond: float = 1e-10) -> T2EQResult:
    """Run the T2EQ equivalence test for two sets of dissolution profiles.

    Parameters
    ----------
    X : array-like, shape (m, p)
        Reference (or pre-change) sample. Rows are units (tablets), columns are time points.
    Y : array-like, shape (n, p)
        Test (or post-change) sample. Same structure as X. Number of columns p must match.
    alpha : float, default 0.05
        Significance level.
    D : array-like, shape (p,), optional
        Acceptance vector (defaults to 10 at each time point, i.e., EMA/FDA 10% rule). Used to build
        the internal equivalence margin as EM = D' S^{-1} D.
    use_pinv_if_needed : bool, default True
        If True, uses Moore-Penrose pseudoinverse when the pooled covariance matrix is singular.
    rcond : float, default 1e-10
        rcond for pseudoinverse.

    Returns
    -------
    T2EQResult
        Dataclass containing p-value, decision (True = equivalent), and rich diagnostics.

    Notes
    -----
    The test follows the CRAN T2EQ implementation (Hoffelder variant for dissolution data):

        S_pooled = ((m-1) S_X + (n-1) S_Y) / (N - 2)
        MD = (R - T)' S_pooled^{-1} (R - T)
        T^2 = (m*n/N) * MD
        EM = D' S_pooled^{-1} D, with D=(10,...,10)
        ncp = (m*n/N) * EM
        df1 = p, df2 = N - 1 - p
        test_stat = T^2 * (N - 1 - p) / ((N - 2) * p)
        crit_value = F^{-1}_{df1, df2; ncp}(alpha)  # left-tail quantile of non-central F
        p_value = F_{df1, df2; ncp}(test_stat)      # left-tail p-value
        decision: p_value <= alpha  (equivalently test_stat < crit_value)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays: (units, timepoints)")
    m, pX = X.shape
    n, pY = Y.shape
    if pX != pY:
        raise ValueError(f"X and Y must have the same number of columns (time points): {pX} != {pY}")
    p = pX
    N = m + n
    if N - 1 - p <= 0:
        raise ValueError(f"Degrees of freedom df2 = N - 1 - p must be positive; got N={N}, p={p}.")

    mean_ref = X.mean(axis=0)
    mean_test = Y.mean(axis=0)
    S_ref = _cov_ddof1(X)
    S_test = _cov_ddof1(Y)
    S_pooled = (((m - 1) * S_ref) + ((n - 1) * S_test)) / (N - 2)

    # Inverse (robust)
    used_pinv = False
    try:
        S_inv = np.linalg.inv(S_pooled)
    except np.linalg.LinAlgError:
        if not use_pinv_if_needed:
            raise
        S_inv = np.linalg.pinv(S_pooled, rcond=rcond)
        used_pinv = True

    delta = mean_ref - mean_test
    MD = float(delta.T @ S_inv @ delta)
    T2 = (m * n / N) * MD

    if D is None:
        D_vec = np.full(p, 10.0)
    else:
        D_vec = np.asarray(D, dtype=float).reshape(-1)
        if D_vec.shape[0] != p:
            raise ValueError(f"D must have length p={p}, got {D_vec.shape[0]}")

    eq_margin = float(D_vec.T @ S_inv @ D_vec)
    ncp = (m * n / N) * eq_margin

    df1 = p
    df2 = N - 1 - p
    numerator = N - 1 - p
    denominator = (N - 2) * p
    test_stat = T2 * numerator / denominator

    # Quantile and CDF of non-central F
    if _HAVE_SCIPY:
        crit_value = float(ncf.ppf(alpha, df1, df2, ncp))
        p_value = float(ncf.cdf(test_stat, df1, df2, ncp))
    else:  # fallback to mpmath implementation
        cdf_val, ppf_val = _ncf_cdf_ppf_with_mpmath(x=test_stat, q=alpha, df1=df1, df2=df2, ncp=ncp)
        crit_value = float(ppf_val)
        p_value = float(cdf_val)

    decision = bool(test_stat < crit_value)

    return T2EQResult(
        p_value=p_value,
        decision=decision,
        alpha=alpha,
        T2=float(T2),
        MD=float(MD),
        test_stat=float(test_stat),
        crit_value=float(crit_value),
        ncp=float(ncp),
        df1=df1,
        df2=df2,
        m=m,
        n=n,
        p=p,
        N=N,
        mean_ref=mean_ref,
        mean_test=mean_test,
        S_ref=S_ref,
        S_test=S_test,
        S_pooled=S_pooled,
        eq_margin=float(eq_margin),
        D=D_vec,
        used_pinv=used_pinv,
    )


def example_usage(seed: int = 123) -> Dict[str, Any]:
    """Generate a small reproducible example with synthetic dissolution profiles and run T2EQ.

    Returns a dictionary with the result dataclass and the input arrays.
    """
    rng = np.random.default_rng(seed)
    p = 5  # time points
    m = 12
    n = 12
    # Simulate correlated dissolution % with moderate covariance
    base_mean = np.array([20, 45, 65, 80, 92], dtype=float)
    L = np.tril(np.ones((p, p)))  # induce correlation
    Sigma = (5.0 * L @ L.T)  # positive-definite

    X = rng.multivariate_normal(base_mean, Sigma, size=m)
    # Slight shift in TEST
    Y = rng.multivariate_normal(base_mean + np.array([1, -1, 0.5, 0.5, 0.2]), Sigma, size=n)

    res = t2eq_test(X, Y, alpha=0.05)
    return {
        "X": X,
        "Y": Y,
        "result": res,
    }


if __name__ == "__main__":
    demo = example_usage()
    r = demo["result"]
    print("T2EQ p-value:", r.p_value)
    print("Decision (True=equivalent):", r.decision)
    print("T2:", r.T2, "MD:", r.MD)
    print("test_stat:", r.test_stat, "crit_value:", r.crit_value)
    print("ncp:", r.ncp, "df1:", r.df1, "df2:", r.df2)
