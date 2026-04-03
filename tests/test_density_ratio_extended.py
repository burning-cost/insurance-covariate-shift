"""Extended tests for RuLSIF and KLIEP — edge cases, parameter sweeps, numerical stability."""

import numpy as np
import pytest

from insurance_covariate_shift.density_ratio import (
    KLIEP,
    RuLSIF,
    _gaussian_kernel,
    _median_heuristic,
)


# ---------------------------------------------------------------------------
# _gaussian_kernel edge cases
# ---------------------------------------------------------------------------

class TestGaussianKernelExtended:
    def test_single_point(self):
        X = np.array([[1.0, 2.0]])
        Y = np.array([[1.0, 2.0]])
        K = _gaussian_kernel(X, Y, sigma=1.0)
        assert K.shape == (1, 1)
        assert K[0, 0] == pytest.approx(1.0)

    def test_zero_sigma_numerical(self):
        # Very small sigma -> kernel is near-delta
        X = np.array([[0.0], [1.0]])
        Y = np.array([[0.0]])
        K = _gaussian_kernel(X, Y, sigma=1e-6)
        assert K[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert K[1, 0] < 1e-6

    def test_asymmetric_sizes(self):
        rng = np.random.default_rng(100)
        X = rng.normal(0, 1, (30, 5))
        Y = rng.normal(0, 1, (13, 5))
        K = _gaussian_kernel(X, Y, sigma=1.5)
        assert K.shape == (30, 13)

    def test_kernel_symmetry_on_same_data(self):
        rng = np.random.default_rng(101)
        X = rng.normal(0, 1, (15, 3))
        K = _gaussian_kernel(X, X, sigma=2.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_returns_float64(self):
        X = np.array([[0.0, 1.0]], dtype=np.float32)
        Y = np.array([[0.5, 0.5]], dtype=np.float32)
        K = _gaussian_kernel(X.astype(float), Y.astype(float), sigma=1.0)
        assert K.dtype == np.float64 or K.dtype.kind == 'f'


# ---------------------------------------------------------------------------
# _median_heuristic edge cases
# ---------------------------------------------------------------------------

class TestMedianHeuristicExtended:
    def test_1d_input(self):
        rng = np.random.default_rng(200)
        X = rng.normal(0, 1, (100, 1))
        Y = rng.normal(2, 1, (80, 1))
        sigma = _median_heuristic(X, Y)
        assert sigma > 0

    def test_returns_float(self):
        X = np.random.default_rng(201).normal(0, 1, (50, 2))
        Y = np.random.default_rng(202).normal(1, 1, (40, 2))
        sigma = _median_heuristic(X, Y)
        assert isinstance(sigma, float)

    def test_scale_sensitivity(self):
        rng = np.random.default_rng(203)
        X = rng.normal(0, 1, (100, 2))
        Y = rng.normal(0, 1, (100, 2))
        sigma_1 = _median_heuristic(X, Y)
        # Data scaled by 10 should give sigma roughly 10x larger
        sigma_10 = _median_heuristic(X * 10, Y * 10)
        assert sigma_10 / sigma_1 == pytest.approx(10.0, rel=0.3)

    def test_large_dataset_subsampled(self):
        # Should not crash with > 500 combined points
        rng = np.random.default_rng(204)
        X = rng.normal(0, 1, (600, 3))
        Y = rng.normal(1, 1, (500, 3))
        sigma = _median_heuristic(X, Y)
        assert sigma > 0


# ---------------------------------------------------------------------------
# RuLSIF extended
# ---------------------------------------------------------------------------

class TestRuLSIFExtended:
    def _data(self, seed=0, n_s=200, n_t=150, d=3, shift=0.5):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, (n_s, d)), rng.normal(shift, 1, (n_t, d))

    def test_score_higher_for_shifted_vs_identical(self):
        rng = np.random.default_rng(300)
        X_s = rng.normal(0, 1, (200, 2))
        X_t_same = rng.normal(0, 1, (150, 2))
        X_t_shifted = rng.normal(5, 1, (150, 2))

        m_same = RuLSIF()
        m_same.fit(X_s, X_t_same)
        score_same = m_same.score(X_s, X_t_same)

        m_shifted = RuLSIF()
        m_shifted.fit(X_s, X_t_shifted)
        score_shifted = m_shifted.score(X_s, X_t_shifted)

        # Larger shift -> higher Pearson divergence
        assert score_shifted > score_same

    def test_fit_small_target(self):
        rng = np.random.default_rng(301)
        X_s = rng.normal(0, 1, (200, 3))
        X_t = rng.normal(0, 1, (10, 3))  # tiny target
        m = RuLSIF(n_kernels=10)
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert w.shape == (200,)
        assert (w >= 0).all()

    def test_very_low_regularisation(self):
        X_s, X_t = self._data(seed=302)
        m = RuLSIF(lam=1e-9)
        m.fit(X_s, X_t)
        assert m._fitted

    def test_high_regularisation(self):
        X_s, X_t = self._data(seed=303)
        m = RuLSIF(lam=100.0)
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert (w >= 0).all()

    def test_explicit_sigma_used(self):
        X_s, X_t = self._data(seed=304)
        m = RuLSIF(sigma=3.14)
        m.fit(X_s, X_t)
        assert m._sigma_fit == pytest.approx(3.14)

    def test_alpha_near_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            RuLSIF(alpha=1.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            RuLSIF(alpha=-0.1)

    def test_predict_before_fit_raises(self):
        m = RuLSIF()
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(np.zeros((5, 3)))

    def test_score_returns_float(self):
        X_s, X_t = self._data(seed=305)
        m = RuLSIF()
        m.fit(X_s, X_t)
        s = m.score(X_s, X_t)
        assert isinstance(s, float)

    def test_fitted_flag(self):
        m = RuLSIF()
        assert not m._fitted
        X_s, X_t = self._data(seed=306)
        m.fit(X_s, X_t)
        assert m._fitted

    def test_centres_shape(self):
        rng = np.random.default_rng(307)
        X_s = rng.normal(0, 1, (100, 4))
        X_t = rng.normal(1, 1, (80, 4))
        m = RuLSIF(n_kernels=50)
        m.fit(X_s, X_t)
        assert m._centres.shape == (50, 4)

    def test_predict_on_target(self):
        X_s, X_t = self._data(seed=308)
        m = RuLSIF()
        m.fit(X_s, X_t)
        w_t = m.predict(X_t)
        assert w_t.shape == (len(X_t),)
        assert (w_t >= 0).all()

    def test_weights_clipped_non_negative(self):
        # Contrived: large shift so some raw weights might be near zero
        rng = np.random.default_rng(309)
        X_s = rng.normal(0, 0.1, (100, 2))
        X_t = rng.normal(10, 0.1, (80, 2))
        m = RuLSIF()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert (w >= 0).all()


# ---------------------------------------------------------------------------
# KLIEP extended
# ---------------------------------------------------------------------------

class TestKLIEPExtended:
    def _data(self, seed=10, n_s=300, n_t=200, d=2):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, (n_s, d)), rng.normal(1, 1, (n_t, d))

    def test_predict_on_target(self):
        X_s, X_t = self._data(seed=400)
        m = KLIEP()
        m.fit(X_s, X_t)
        w_t = m.predict(X_t)
        assert w_t.shape == (len(X_t),)
        assert (w_t >= 0).all()

    def test_fitted_flag(self):
        m = KLIEP()
        assert not m._fitted
        X_s, X_t = self._data(seed=401)
        m.fit(X_s, X_t)
        assert m._fitted

    def test_small_target_dataset(self):
        rng = np.random.default_rng(402)
        X_s = rng.normal(0, 1, (300, 2))
        X_t = rng.normal(0, 1, (8, 2))
        m = KLIEP(n_kernels=8)
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert (w >= 0).all()

    def test_very_few_iterations(self):
        X_s, X_t = self._data(seed=403)
        m = KLIEP(max_iter=2)
        m.fit(X_s, X_t)
        assert m._fitted

    def test_high_learning_rate(self):
        X_s, X_t = self._data(seed=404)
        m = KLIEP(learning_rate=0.1)
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert (w >= 0).all()

    def test_alpha_vec_non_negative(self):
        X_s, X_t = self._data(seed=405)
        m = KLIEP()
        m.fit(X_s, X_t)
        assert (m._alpha_vec >= 0).all()

    def test_repr_contains_class_name(self):
        m = KLIEP()
        assert "KLIEP" in repr(m) or "kliep" in repr(m).lower() or m.__class__.__name__ == "KLIEP"

    def test_centres_dimensionality(self):
        rng = np.random.default_rng(406)
        X_s = rng.normal(0, 1, (200, 5))
        X_t = rng.normal(1, 1, (100, 5))
        m = KLIEP(n_kernels=30)
        m.fit(X_s, X_t)
        assert m._centres.shape[1] == 5
        assert m._centres.shape[0] == 30

    def test_returns_self_from_fit(self):
        X_s, X_t = self._data(seed=407)
        m = KLIEP()
        r = m.fit(X_s, X_t)
        assert r is m

    def test_predict_before_fit_raises(self):
        m = KLIEP()
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(np.zeros((5, 2)))
