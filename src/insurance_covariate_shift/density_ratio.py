"""
Density ratio estimation methods: RuLSIF and KLIEP.

These are the closed-form / iterative alternatives to the CatBoost classifier
approach. RuLSIF is preferred when features are all continuous (e.g. modelled
scores, rating factors). KLIEP is the reference algorithm and handles the
normalisation constraint explicitly.

References
----------
- Yamada et al. (2013) "Relative Density-Ratio Estimation for Robust
  Distribution Comparison". Neural Computation 25(5).
- Sugiyama et al. (2008) "Direct Importance Estimation with Model Selection
  and Its Application to Covariate Shift Adaptation". NeurIPS.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


__all__ = ["RuLSIF", "KLIEP"]


def _gaussian_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """
    Gaussian (RBF) kernel matrix K[i,j] = exp(-||x_i - y_j||^2 / (2 sigma^2)).

    Parameters
    ----------
    X : array of shape (n, d)
    Y : array of shape (m, d)
    sigma : bandwidth

    Returns
    -------
    K : array of shape (n, m)
    """
    sq_dists = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def _median_heuristic(X: NDArray[np.float64], Y: NDArray[np.float64]) -> float:
    """
    Median heuristic for Gaussian kernel bandwidth.

    Computes the median pairwise distance across both sets combined and uses
    that as the bandwidth. Standard practice in kernel methods; works well
    for moderate dimensionality.
    """
    combined = np.vstack([X, Y])
    n = min(500, len(combined))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(combined), size=n, replace=False)
    sub = combined[idx]
    sq_dists = cdist(sub, sub, metric="sqeuclidean")
    # upper triangle only to avoid double counting
    upper = sq_dists[np.triu_indices(n, k=1)]
    median_sq = np.median(upper)
    if median_sq < 1e-12:
        return 1.0
    return float(np.sqrt(median_sq))


class RuLSIF:
    """
    Relative Unconstrained Least-Squares Importance Fitting (RuLSIF).

    Estimates the alpha-relative density ratio:

        r_alpha(x) = p_target(x) / (alpha * p_target(x) + (1 - alpha) * p_source(x))

    The alpha parameter prevents extreme weights when the two distributions
    barely overlap — set alpha=0 for ordinary density ratio (uLSIF), alpha>0
    for the relative version which is more numerically stable.

    The solution is closed-form:

        theta* = (H + lambda * I)^{-1} h

    where H and h are kernel matrix aggregates. This makes RuLSIF very fast
    even on large datasets.

    Parameters
    ----------
    sigma : float or None
        Gaussian kernel bandwidth. None triggers the median heuristic.
    alpha : float
        Relative ratio parameter in [0, 1). Default 0.1.
    lam : float
        Regularisation strength. Default 1e-3.
    n_kernels : int
        Number of kernel centres (sampled from target). Default 200.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_covariate_shift.density_ratio import RuLSIF
    >>> rng = np.random.default_rng(0)
    >>> X_s = rng.normal(0, 1, (200, 3))
    >>> X_t = rng.normal(0.5, 1, (150, 3))
    >>> model = RuLSIF()
    >>> model.fit(X_s, X_t)
    >>> w = model.predict(X_s)
    >>> w.shape
    (200,)
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        alpha: float = 0.1,
        lam: float = 1e-3,
        n_kernels: int = 200,
    ) -> None:
        if not (0.0 <= alpha < 1.0):
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self.sigma = sigma
        self.alpha = alpha
        self.lam = lam
        self.n_kernels = n_kernels

        self._sigma_fit: float = 1.0
        self._centres: NDArray[np.float64] = np.empty((0, 0))
        self._theta: NDArray[np.float64] = np.empty(0)
        self._fitted = False

    def fit(
        self,
        X_source: NDArray[np.float64],
        X_target: NDArray[np.float64],
    ) -> "RuLSIF":
        """
        Fit the density ratio model.

        Parameters
        ----------
        X_source : array of shape (n_source, d)
        X_target : array of shape (n_target, d)

        Returns
        -------
        self
        """
        X_source = np.asarray(X_source, dtype=float)
        X_target = np.asarray(X_target, dtype=float)

        n_s = len(X_source)
        n_t = len(X_target)
        n_c = min(self.n_kernels, n_t)

        # Kernel centres from target
        rng = np.random.default_rng(42)
        idx = rng.choice(n_t, size=n_c, replace=False)
        self._centres = X_target[idx]

        # Bandwidth
        if self.sigma is None:
            self._sigma_fit = _median_heuristic(X_source, X_target)
        else:
            self._sigma_fit = float(self.sigma)

        # Kernel matrices
        Phi_s = _gaussian_kernel(X_source, self._centres, self._sigma_fit)  # (n_s, n_c)
        Phi_t = _gaussian_kernel(X_target, self._centres, self._sigma_fit)  # (n_t, n_c)

        # H = alpha/n_t * Phi_t^T Phi_t + (1-alpha)/n_s * Phi_s^T Phi_s
        H = (
            self.alpha / n_t * Phi_t.T @ Phi_t
            + (1.0 - self.alpha) / n_s * Phi_s.T @ Phi_s
        )
        # h = (1/n_t) * sum_i phi(x_t_i)
        h = Phi_t.mean(axis=0)

        # Closed-form solution with Tikhonov regularisation
        A = H + self.lam * np.eye(n_c)
        try:
            self._theta = np.linalg.solve(A, h)
        except np.linalg.LinAlgError:
            self._theta = np.linalg.lstsq(A, h, rcond=None)[0]

        self._fitted = True
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict density ratio weights for new points.

        Weights are clipped to be non-negative — RuLSIF has no built-in
        non-negativity guarantee.

        Parameters
        ----------
        X : array of shape (n, d)

        Returns
        -------
        weights : array of shape (n,)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        Phi = _gaussian_kernel(X, self._centres, self._sigma_fit)
        w = Phi @ self._theta
        return np.maximum(w, 0.0)

    def score(
        self,
        X_source: NDArray[np.float64],
        X_target: NDArray[np.float64],
    ) -> float:
        """
        Pearson divergence estimate (objective value, higher = more distinct).

        The RuLSIF objective is: (1/2) * E_target[r(x)^2] - E_source[r(x)].
        The first term uses target predictions; the second uses source.
        """
        w_t = self.predict(X_target)
        w_s = self.predict(X_source)
        return float(0.5 * np.mean(w_t ** 2) - np.mean(w_s))


class KLIEP:
    """
    Kullback-Leibler Importance Estimation Procedure (KLIEP).

    Directly minimises KL(p_target || p_source * w) via projected gradient
    ascent with a normalisation constraint: E_source[w(x)] = 1.

    KLIEP is the canonical reference method. It handles the constraint
    explicitly and produces weights that are unbiased in expectation. It is
    slower than RuLSIF but is a useful sanity-check and preferred when the
    normalisation property matters for exposure weighting.

    Parameters
    ----------
    sigma : float or None
        Gaussian kernel bandwidth. None triggers the median heuristic.
    n_kernels : int
        Number of basis functions (kernel centres from target).
    max_iter : int
        Maximum gradient ascent iterations.
    learning_rate : float
        Step size for gradient ascent.
    tol : float
        Convergence tolerance on the objective.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_covariate_shift.density_ratio import KLIEP
    >>> rng = np.random.default_rng(1)
    >>> X_s = rng.normal(0, 1, (300, 2))
    >>> X_t = rng.normal(1, 1, (200, 2))
    >>> model = KLIEP()
    >>> model.fit(X_s, X_t)
    >>> w = model.predict(X_s)
    >>> abs(w.mean() - 1.0) < 0.1  # normalisation
    True
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        n_kernels: int = 200,
        max_iter: int = 1000,
        learning_rate: float = 1e-3,
        tol: float = 1e-6,
    ) -> None:
        self.sigma = sigma
        self.n_kernels = n_kernels
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol

        self._sigma_fit: float = 1.0
        self._centres: NDArray[np.float64] = np.empty((0, 0))
        self._alpha_vec: NDArray[np.float64] = np.empty(0)
        self._fitted = False

    def _w_raw(self, Phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """w(x) = sum_l alpha_l * k(x, c_l), always >= 0 because alpha >= 0."""
        return Phi @ self._alpha_vec

    def _normalise(self, Phi_s: NDArray[np.float64]) -> None:
        """Project alpha so that mean_source[w(x)] = 1."""
        denom = float(self._w_raw(Phi_s).mean())
        if denom > 1e-12:
            self._alpha_vec /= denom

    def fit(
        self,
        X_source: NDArray[np.float64],
        X_target: NDArray[np.float64],
    ) -> "KLIEP":
        """
        Fit using projected gradient ascent on log E_target[log w(x)].

        Parameters
        ----------
        X_source : array of shape (n_source, d)
        X_target : array of shape (n_target, d)

        Returns
        -------
        self
        """
        X_source = np.asarray(X_source, dtype=float)
        X_target = np.asarray(X_target, dtype=float)

        n_t = len(X_target)
        n_c = min(self.n_kernels, n_t)

        rng = np.random.default_rng(0)
        idx = rng.choice(n_t, size=n_c, replace=False)
        self._centres = X_target[idx]

        if self.sigma is None:
            self._sigma_fit = _median_heuristic(X_source, X_target)
        else:
            self._sigma_fit = float(self.sigma)

        Phi_s = _gaussian_kernel(X_source, self._centres, self._sigma_fit)
        Phi_t = _gaussian_kernel(X_target, self._centres, self._sigma_fit)

        # Initialise alpha uniformly, then normalise
        self._alpha_vec = np.ones(n_c) / n_c
        self._normalise(Phi_s)

        prev_obj = -np.inf
        for _ in range(self.max_iter):
            w_t = self._w_raw(Phi_t)
            # Avoid log(0)
            w_t = np.maximum(w_t, 1e-12)
            # Gradient: (1/n_t) * sum_i Phi_t[i] / w_t[i]
            grad = (Phi_t / w_t[:, None]).mean(axis=0)
            self._alpha_vec = np.maximum(self._alpha_vec + self.learning_rate * grad, 0.0)
            self._normalise(Phi_s)

            obj = float(np.mean(np.log(np.maximum(self._w_raw(Phi_t), 1e-12))))
            if abs(obj - prev_obj) < self.tol:
                break
            prev_obj = obj

        self._fitted = True
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict density ratio weights.

        Parameters
        ----------
        X : array of shape (n, d)

        Returns
        -------
        weights : array of shape (n,)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        Phi = _gaussian_kernel(X, self._centres, self._sigma_fit)
        return np.maximum(self._w_raw(Phi), 0.0)
