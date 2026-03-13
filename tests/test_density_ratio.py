"""Tests for density ratio estimation: RuLSIF and KLIEP."""

import numpy as np
import pytest

from insurance_covariate_shift.density_ratio import (
    KLIEP,
    RuLSIF,
    _gaussian_kernel,
    _median_heuristic,
)


RNG = np.random.default_rng(42)


# -----------------------------------------------------------------------
# Gaussian kernel
# -----------------------------------------------------------------------

class TestGaussianKernel:
    def test_shape(self):
        X = RNG.normal(0, 1, (10, 3))
        Y = RNG.normal(0, 1, (7, 3))
        K = _gaussian_kernel(X, Y, sigma=1.0)
        assert K.shape == (10, 7)

    def test_diagonal_self_similarity(self):
        X = RNG.normal(0, 1, (5, 2))
        K = _gaussian_kernel(X, X, sigma=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_values_between_zero_and_one(self):
        X = RNG.normal(0, 1, (20, 4))
        Y = RNG.normal(0, 1, (15, 4))
        K = _gaussian_kernel(X, Y, sigma=2.0)
        assert (K >= 0).all()
        assert (K <= 1.0 + 1e-10).all()

    def test_large_sigma_approaches_one(self):
        X = RNG.normal(0, 1, (10, 2))
        Y = RNG.normal(0, 1, (10, 2))
        K = _gaussian_kernel(X, Y, sigma=1000.0)
        np.testing.assert_allclose(K, 1.0, atol=0.01)

    def test_small_sigma_approaches_zero_off_diagonal(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        Y = np.array([[0.0, 0.0]])
        K = _gaussian_kernel(X, Y, sigma=0.1)
        assert K[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert K[1, 0] < 1e-6


class TestMedianHeuristic:
    def test_returns_positive(self):
        X = RNG.normal(0, 1, (50, 3))
        Y = RNG.normal(1, 1, (40, 3))
        sigma = _median_heuristic(X, Y)
        assert sigma > 0

    def test_identical_data_fallback(self):
        # All-zero data — pairwise distances are zero — should return 1.0
        X = np.zeros((20, 3))
        Y = np.zeros((15, 3))
        sigma = _median_heuristic(X, Y)
        assert sigma == 1.0


# -----------------------------------------------------------------------
# RuLSIF
# -----------------------------------------------------------------------

class TestRuLSIF:
    def _make_data(self):
        rng = np.random.default_rng(0)
        X_s = rng.normal(0, 1, (200, 3))
        X_t = rng.normal(0.5, 1, (150, 3))
        return X_s, X_t

    def test_fit_predict_shape(self):
        X_s, X_t = self._make_data()
        m = RuLSIF()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert w.shape == (200,)

    def test_weights_non_negative(self):
        X_s, X_t = self._make_data()
        m = RuLSIF()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert (w >= 0).all()

    def test_returns_self(self):
        X_s, X_t = self._make_data()
        m = RuLSIF()
        result = m.fit(X_s, X_t)
        assert result is m

    def test_unfitted_raises(self):
        m = RuLSIF()
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(np.zeros((5, 3)))

    def test_identical_distributions_weights_near_one(self):
        rng = np.random.default_rng(1)
        X_s = rng.normal(0, 1, (300, 2))
        X_t = rng.normal(0, 1, (300, 2))
        m = RuLSIF()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        # With identical distributions weights should be centred near 1
        assert 0.5 < w.mean() < 2.0

    def test_shifted_distribution_has_varied_weights(self):
        rng = np.random.default_rng(2)
        X_s = rng.normal(0, 1, (300, 2))
        X_t = rng.normal(3, 1, (200, 2))  # Large shift
        m = RuLSIF()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        # Weights should vary substantially
        assert w.std() > 0.1

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            RuLSIF(alpha=1.0)

    def test_alpha_zero(self):
        X_s, X_t = self._make_data()
        m = RuLSIF(alpha=0.0)
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert w.shape == (200,)

    def test_custom_sigma(self):
        X_s, X_t = self._make_data()
        m = RuLSIF(sigma=2.0)
        m.fit(X_s, X_t)
        assert m._sigma_fit == 2.0

    def test_score_method(self):
        X_s, X_t = self._make_data()
        m = RuLSIF()
        m.fit(X_s, X_t)
        s = m.score(X_s, X_t)
        assert isinstance(s, float)

    def test_1d_input(self):
        rng = np.random.default_rng(3)
        X_s = rng.normal(0, 1, (100, 1))
        X_t = rng.normal(1, 1, (80, 1))
        m = RuLSIF()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert w.shape == (100,)

    def test_n_kernels_capped_at_n_target(self):
        rng = np.random.default_rng(4)
        X_s = rng.normal(0, 1, (100, 2))
        X_t = rng.normal(0, 1, (50, 2))  # 50 < default 200
        m = RuLSIF(n_kernels=200)
        m.fit(X_s, X_t)
        assert m._centres.shape[0] == 50

    def test_high_dimensional(self):
        rng = np.random.default_rng(5)
        X_s = rng.normal(0, 1, (150, 20))
        X_t = rng.normal(0.3, 1, (100, 20))
        m = RuLSIF()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert (w >= 0).all()


# -----------------------------------------------------------------------
# KLIEP
# -----------------------------------------------------------------------

class TestKLIEP:
    def _make_data(self):
        rng = np.random.default_rng(10)
        X_s = rng.normal(0, 1, (300, 2))
        X_t = rng.normal(1, 1, (200, 2))
        return X_s, X_t

    def test_fit_predict_shape(self):
        X_s, X_t = self._make_data()
        m = KLIEP()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert w.shape == (300,)

    def test_weights_non_negative(self):
        X_s, X_t = self._make_data()
        m = KLIEP()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert (w >= 0).all()

    def test_returns_self(self):
        X_s, X_t = self._make_data()
        m = KLIEP()
        result = m.fit(X_s, X_t)
        assert result is m

    def test_unfitted_raises(self):
        m = KLIEP()
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(np.zeros((5, 2)))

    def test_normalisation_constraint(self):
        # E_source[w(x)] should be approximately 1
        rng = np.random.default_rng(11)
        X_s = rng.normal(0, 1, (400, 2))
        X_t = rng.normal(0.5, 1, (300, 2))
        m = KLIEP()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        # Mean should be close to 1 (normalisation constraint)
        assert abs(w.mean() - 1.0) < 0.2

    def test_identical_distributions(self):
        rng = np.random.default_rng(12)
        X_s = rng.normal(0, 1, (300, 2))
        X_t = rng.normal(0, 1, (300, 2))
        m = KLIEP()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert abs(w.mean() - 1.0) < 0.3

    def test_shifted_distribution(self):
        rng = np.random.default_rng(13)
        X_s = rng.normal(0, 1, (300, 2))
        X_t = rng.normal(2, 1, (200, 2))
        m = KLIEP()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert w.std() > 0.1

    def test_custom_sigma(self):
        X_s, X_t = self._make_data()
        m = KLIEP(sigma=1.5)
        m.fit(X_s, X_t)
        assert m._sigma_fit == 1.5

    def test_n_kernels_capped(self):
        rng = np.random.default_rng(14)
        X_s = rng.normal(0, 1, (100, 2))
        X_t = rng.normal(0, 1, (30, 2))
        m = KLIEP(n_kernels=200)
        m.fit(X_s, X_t)
        assert m._centres.shape[0] == 30

    def test_convergence_with_tolerance(self):
        X_s, X_t = self._make_data()
        m = KLIEP(tol=1e-3, max_iter=200)
        m.fit(X_s, X_t)
        assert m._fitted

    def test_1d_input(self):
        rng = np.random.default_rng(15)
        X_s = rng.normal(0, 1, (200, 1))
        X_t = rng.normal(1, 1, (150, 1))
        m = KLIEP()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert w.shape == (200,)
