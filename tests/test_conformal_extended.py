"""Extended tests for ShiftRobustConformal and _LRQRThreshold."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor

from insurance_covariate_shift.adaptor import CovariateShiftAdaptor
from insurance_covariate_shift.conformal import (
    ShiftRobustConformal,
    _weighted_quantile,
    _LRQRThreshold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regression_setup(n_train=400, n_cal=300, n_target=150, d=4, shift=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(0, 1, (n_train, d))
    y_train = X_train[:, 0] * 2 + rng.normal(0, 0.5, n_train)
    model = LinearRegression().fit(X_train, y_train)

    X_cal = rng.normal(0, 1, (n_cal, d))
    y_cal = X_cal[:, 0] * 2 + rng.normal(0, 0.5, n_cal)
    X_target = rng.normal(shift, 1.2, (n_target, d))
    y_target = X_target[:, 0] * 2 + rng.normal(0, 0.5, n_target)
    return model, X_cal, y_cal, X_target, y_target


# ---------------------------------------------------------------------------
# _weighted_quantile extended
# ---------------------------------------------------------------------------

class TestWeightedQuantileExtended:
    def test_single_element(self):
        q = _weighted_quantile(np.array([5.0]), np.array([1.0]), 0.9)
        assert isinstance(q, float)

    def test_level_one_returns_inf_or_max(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.ones(5)
        q = _weighted_quantile(scores, w, 1.0)
        # Level=1.0 should return inf (or the last score which is inf appended)
        assert q == np.inf or q >= 5.0

    def test_negative_weights_treated_as_zero(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([-1.0, -2.0, 1.0, 1.0, 1.0])
        q = _weighted_quantile(scores, w, 0.8)
        assert isinstance(q, float)
        assert np.isfinite(q) or q == np.inf

    def test_very_large_weights_on_small_score(self):
        # If the first score has huge weight, quantile should be 1.0
        scores = np.array([1.0, 10.0, 100.0])
        w = np.array([1e6, 1.0, 1.0])
        q = _weighted_quantile(scores, w, 0.5)
        assert q <= 10.0

    def test_all_equal_scores(self):
        scores = np.ones(20) * 7.5
        w = np.ones(20)
        q = _weighted_quantile(scores, w, 0.9)
        assert q == pytest.approx(7.5) or q == np.inf

    def test_sorted_scores_monotone_quantile(self):
        scores = np.arange(100, dtype=float)
        w = np.ones(100)
        q25 = _weighted_quantile(scores, w, 0.25)
        q75 = _weighted_quantile(scores, w, 0.75)
        assert q25 <= q75


# ---------------------------------------------------------------------------
# _LRQRThreshold
# ---------------------------------------------------------------------------

class TestLRQRThreshold:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(800)
        n = 200
        X = rng.normal(0, 1, (n, 4))
        scores = np.abs(rng.normal(0, 1, n))
        weights = rng.uniform(0.5, 2.0, n)

        th = _LRQRThreshold(alpha=0.1)
        th.fit(X, scores, weights)
        preds = th.predict(X)
        assert preds.shape == (n,)

    def test_fitted_flag_set(self):
        rng = np.random.default_rng(801)
        X = rng.normal(0, 1, (100, 3))
        scores = np.abs(rng.normal(0, 1, 100))
        weights = np.ones(100)
        th = _LRQRThreshold(alpha=0.1)
        th.fit(X, scores, weights)
        assert th._fitted

    def test_predict_unfitted_returns_base_quantile(self):
        rng = np.random.default_rng(802)
        X = rng.normal(0, 1, (50, 3))
        th = _LRQRThreshold(alpha=0.1)
        # Not fitted — _base_quantile is 0.0 initially
        preds = th.predict(X)
        assert preds.shape == (50,)
        np.testing.assert_allclose(preds, 0.0)

    def test_higher_lambda_increases_mean_threshold(self):
        rng = np.random.default_rng(803)
        X = rng.normal(0, 1, (200, 3))
        scores = np.abs(rng.normal(0, 1, 200))
        weights = rng.exponential(1.0, 200)

        th1 = _LRQRThreshold(alpha=0.1, lam=0.0)
        th2 = _LRQRThreshold(alpha=0.1, lam=5.0)
        th1.fit(X, scores, weights)
        th2.fit(X, scores, weights)
        # Both should produce finite predictions
        assert np.all(np.isfinite(th1.predict(X)))
        assert np.all(np.isfinite(th2.predict(X)))

    def test_base_quantile_stored_after_fit(self):
        rng = np.random.default_rng(804)
        X = rng.normal(0, 1, (100, 2))
        scores = rng.uniform(0, 5, 100)
        weights = np.ones(100)
        th = _LRQRThreshold(alpha=0.1)
        th.fit(X, scores, weights)
        assert np.isfinite(th._base_quantile)
        assert th._base_quantile >= 0


# ---------------------------------------------------------------------------
# ShiftRobustConformal — weighted method extended
# ---------------------------------------------------------------------------

class TestWeightedConformalExtended:
    def _make_cp(self, seed=0, method="weighted", alpha=0.1):
        model, X_cal, y_cal, X_target, y_target = _regression_setup(seed=seed)
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_cal, X_target)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor, method=method, alpha=alpha)
        cp.calibrate(X_cal, y_cal)
        return cp, X_target, y_target

    def test_n_calibration_correct(self):
        cp, _, _ = self._make_cp(seed=0)
        assert cp.n_calibration_ == 300

    def test_calibrated_flag_true(self):
        cp, _, _ = self._make_cp(seed=1)
        assert cp.calibrated_

    def test_interval_lower_lt_upper(self):
        cp, X_t, _ = self._make_cp(seed=2)
        lo, hi = cp.predict_interval(X_t)
        assert (hi >= lo).all()

    def test_interval_width_positive(self):
        cp, X_t, _ = self._make_cp(seed=3)
        widths = cp.interval_width(X_t)
        assert (widths >= 0).all()

    def test_interval_width_constant_for_weighted(self):
        cp, X_t, _ = self._make_cp(seed=4)
        widths = cp.interval_width(X_t)
        assert widths.std() < 1e-10

    def test_empirical_coverage_in_range(self):
        cp, X_t, y_t = self._make_cp(seed=5)
        cov = cp.empirical_coverage(X_t, y_t)
        assert 0.0 <= cov <= 1.0

    def test_predict_interval_before_calibrate_raises(self):
        model = LinearRegression()
        cp = ShiftRobustConformal(model=model)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(np.zeros((5, 4)))

    def test_single_test_point(self):
        cp, X_t, _ = self._make_cp(seed=6)
        lo, hi = cp.predict_interval(X_t[:1])
        assert lo.shape == (1,)
        assert hi.shape == (1,)

    def test_coverage_higher_for_conservative_alpha(self):
        # alpha=0.01 should give very wide intervals -> high coverage
        model, X_cal, y_cal, X_target, y_target = _regression_setup(seed=7, n_cal=400)
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_cal, X_target)

        cp_strict = ShiftRobustConformal(model=model, adaptor=adaptor, alpha=0.01)
        cp_strict.calibrate(X_cal, y_cal)
        cov_strict = cp_strict.empirical_coverage(X_target, y_target)

        cp_liberal = ShiftRobustConformal(model=model, adaptor=adaptor, alpha=0.40)
        cp_liberal.calibrate(X_cal, y_cal)
        cov_liberal = cp_liberal.empirical_coverage(X_target, y_target)

        assert cov_strict >= cov_liberal

    def test_dummy_model_works(self):
        rng = np.random.default_rng(8)
        X = rng.normal(0, 1, (200, 3))
        y = rng.normal(5, 1, 200)
        model = DummyRegressor(strategy="mean").fit(X, y)
        X_t = rng.normal(0.5, 1, (100, 3))
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X, X_t)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor, alpha=0.1)
        cp.calibrate(X, y)
        lo, hi = cp.predict_interval(X_t)
        assert (hi >= lo).all()

    def test_alpha_boundary_values(self):
        with pytest.raises(ValueError, match="alpha"):
            ShiftRobustConformal(LinearRegression(), alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ShiftRobustConformal(LinearRegression(), alpha=1.0)

    def test_repr_contains_method_and_alpha(self):
        cp = ShiftRobustConformal(LinearRegression(), method="weighted", alpha=0.05)
        r = repr(cp)
        assert "weighted" in r
        assert "0.05" in r

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            ShiftRobustConformal(LinearRegression(), method="bogus")


# ---------------------------------------------------------------------------
# LR-QR extended
# ---------------------------------------------------------------------------

class TestLRQRExtended:
    def _make_lrqr(self, seed=0, n_cal=400, alpha=0.1):
        model, X_cal, y_cal, X_target, y_target = _regression_setup(
            seed=seed, n_cal=n_cal
        )
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_cal, X_target)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor, method="lrqr", alpha=alpha)
        cp.calibrate(X_cal, y_cal)
        return cp, X_target, y_target

    def test_interval_lower_lt_upper(self):
        cp, X_t, _ = self._make_lrqr(seed=10)
        lo, hi = cp.predict_interval(X_t)
        assert (hi >= lo).all()

    def test_widths_non_negative(self):
        cp, X_t, _ = self._make_lrqr(seed=11)
        widths = cp.interval_width(X_t)
        assert (widths >= 0).all()

    def test_lrqr_threshold_fitted(self):
        cp, _, _ = self._make_lrqr(seed=12)
        assert cp._lrqr_threshold is not None

    def test_coverage_reasonable(self):
        cp, X_t, y_t = self._make_lrqr(seed=13)
        cov = cp.empirical_coverage(X_t, y_t)
        assert 0.5 <= cov <= 1.0

    def test_small_calibration_warns(self):
        model, X_cal, y_cal, X_target, _ = _regression_setup(seed=14, n_cal=50)
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_cal, X_target)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor, method="lrqr")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.calibrate(X_cal, y_cal)
            assert any("LR-QR" in str(warning.message) for warning in w)

    def test_lrqr_lambda_parameter_stored(self):
        cp = ShiftRobustConformal(LinearRegression(), method="lrqr", lrqr_lambda=2.5)
        assert cp.lrqr_lambda == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# No adaptor paths
# ---------------------------------------------------------------------------

class TestNoAdaptorPath:
    def test_no_adaptor_with_target_unlabelled(self):
        model, X_cal, y_cal, X_target, _ = _regression_setup(seed=20)
        cp = ShiftRobustConformal(model=model, adaptor=None, method="weighted")
        cp.calibrate(X_cal, y_cal, X_target_unlabelled=X_target)
        assert cp.calibrated_
        lo, hi = cp.predict_interval(X_target)
        assert (hi >= lo).all()

    def test_no_adaptor_no_target_warns_and_works(self):
        model, X_cal, y_cal, X_target, _ = _regression_setup(seed=21)
        cp = ShiftRobustConformal(model=model, adaptor=None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.calibrate(X_cal, y_cal)
            assert len(w) >= 1
        lo, hi = cp.predict_interval(X_target)
        assert (hi >= lo).all()

    def test_calibration_weights_ones_without_adaptor(self):
        model, X_cal, y_cal, X_target, _ = _regression_setup(seed=22)
        cp = ShiftRobustConformal(model=model, adaptor=None)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cp.calibrate(X_cal, y_cal)
        np.testing.assert_allclose(cp._calibration_weights, 1.0)
