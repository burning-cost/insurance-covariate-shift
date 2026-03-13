"""Tests for ShiftRobustConformal."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from insurance_covariate_shift.adaptor import CovariateShiftAdaptor
from insurance_covariate_shift.conformal import ShiftRobustConformal, _weighted_quantile


RNG = np.random.default_rng(42)


def make_regression_problem(n_source=300, n_target=150, shift=0.5, d=4):
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (400, d))
    y_train = X_train[:, 0] * 2 + rng.normal(0, 0.5, 400)
    model = LinearRegression().fit(X_train, y_train)

    X_source = rng.normal(0, 1, (n_source, d))
    y_source = X_source[:, 0] * 2 + rng.normal(0, 0.5, n_source)
    X_target = rng.normal(shift, 1.2, (n_target, d))
    y_target = X_target[:, 0] * 2 + rng.normal(0, 0.5, n_target)
    return model, X_source, y_source, X_target, y_target


# -----------------------------------------------------------------------
# _weighted_quantile
# -----------------------------------------------------------------------

class TestWeightedQuantile:
    def test_uniform_weights_matches_numpy(self):
        rng = np.random.default_rng(1)
        scores = rng.exponential(1, 100)
        weights = np.ones(100)
        wq = _weighted_quantile(scores, weights, 0.9)
        # Should be close to numpy quantile (not identical due to +inf appending)
        assert abs(wq - np.quantile(scores, 0.9)) < 0.5

    def test_zero_weights_fallback(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.zeros(5)
        q = _weighted_quantile(scores, weights, 0.8)
        assert q == pytest.approx(np.quantile(scores, 0.8), abs=1.0)

    def test_level_zero_returns_minimum_or_inf(self):
        scores = np.array([1.0, 2.0, 3.0])
        weights = np.ones(3)
        q = _weighted_quantile(scores, weights, 0.0)
        assert q <= 1.0

    def test_high_weight_inflates_upper_quantile(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        # Heavy weight on last point should pull quantile up
        weights_uniform = np.ones(5)
        weights_heavy = np.array([1.0, 1.0, 1.0, 1.0, 50.0])
        q_uniform = _weighted_quantile(scores, weights_uniform, 0.8)
        q_heavy = _weighted_quantile(scores, weights_heavy, 0.8)
        assert q_heavy >= q_uniform

    def test_returns_float(self):
        scores = np.arange(10, dtype=float)
        weights = np.ones(10)
        assert isinstance(_weighted_quantile(scores, weights, 0.5), float)


# -----------------------------------------------------------------------
# ShiftRobustConformal — init
# -----------------------------------------------------------------------

class TestConformalInit:
    def test_default_method_weighted(self):
        model = LinearRegression()
        cp = ShiftRobustConformal(model)
        assert cp.method == "weighted"

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            ShiftRobustConformal(LinearRegression(), method="bad")

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            ShiftRobustConformal(LinearRegression(), alpha=0.0)

    def test_alpha_one_invalid(self):
        with pytest.raises(ValueError, match="alpha"):
            ShiftRobustConformal(LinearRegression(), alpha=1.0)

    def test_not_calibrated_initially(self):
        cp = ShiftRobustConformal(LinearRegression())
        assert not cp.calibrated_

    def test_repr(self):
        cp = ShiftRobustConformal(LinearRegression(), method="weighted", alpha=0.1)
        r = repr(cp)
        assert "weighted" in r
        assert "0.1" in r


# -----------------------------------------------------------------------
# Weighted conformal
# -----------------------------------------------------------------------

class TestWeightedConformal:
    def _setup(self):
        model, X_s, y_s, X_t, y_t = make_regression_problem()
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_s, X_t)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor, method="weighted", alpha=0.1)
        cp.calibrate(X_s, y_s)
        return cp, X_t, y_t

    def test_calibrate_returns_self(self):
        model, X_s, y_s, X_t, _ = make_regression_problem()
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_s, X_t)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor)
        result = cp.calibrate(X_s, y_s)
        assert result is cp

    def test_calibrated_after_calibrate(self):
        cp, _, _ = self._setup()
        assert cp.calibrated_

    def test_predict_interval_shape(self):
        cp, X_t, _ = self._setup()
        lo, hi = cp.predict_interval(X_t)
        assert lo.shape == (len(X_t),)
        assert hi.shape == (len(X_t),)

    def test_lower_lt_upper(self):
        cp, X_t, _ = self._setup()
        lo, hi = cp.predict_interval(X_t)
        assert (hi > lo).all()

    def test_uncalibrated_predict_raises(self):
        model, X_s, _, X_t, _ = make_regression_problem()
        cp = ShiftRobustConformal(model=model)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(X_t)

    def test_empirical_coverage_is_float(self):
        cp, X_t, y_t = self._setup()
        cov = cp.empirical_coverage(X_t, y_t)
        assert isinstance(cov, float)
        assert 0.0 <= cov <= 1.0

    def test_coverage_roughly_correct(self):
        # With n=150 test points, coverage should be within ~15% of 0.9
        cp, X_t, y_t = self._setup()
        cov = cp.empirical_coverage(X_t, y_t)
        assert 0.7 <= cov <= 1.0

    def test_interval_widths_constant_for_weighted(self):
        cp, X_t, _ = self._setup()
        widths = cp.interval_width(X_t)
        # All widths equal (global quantile)
        assert widths.std() < 1e-10

    def test_n_calibration_stored(self):
        cp, _, _ = self._setup()
        assert cp.n_calibration_ == 300

    def test_no_adaptor_with_target_provided(self):
        model, X_s, y_s, X_t, _ = make_regression_problem()
        cp = ShiftRobustConformal(model=model, adaptor=None, method="weighted")
        cp.calibrate(X_s, y_s, X_target_unlabelled=X_t)
        assert cp.calibrated_
        lo, hi = cp.predict_interval(X_t)
        assert (hi > lo).all()

    def test_no_adaptor_no_target_warns(self):
        model, X_s, y_s, X_t, _ = make_regression_problem()
        cp = ShiftRobustConformal(model=model, adaptor=None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.calibrate(X_s, y_s)
            assert len(w) >= 1

    def test_1d_calibration(self):
        rng = np.random.default_rng(60)
        X = rng.normal(0, 1, (200, 1))
        y = X[:, 0] + rng.normal(0, 0.2, 200)
        model = LinearRegression().fit(X, y)
        cp = ShiftRobustConformal(model=model, method="weighted")
        X_t = rng.normal(0.5, 1, (100, 1))
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X, X_t)
        cp.adaptor = adaptor
        cp.calibrate(X, y)
        lo, hi = cp.predict_interval(X_t)
        assert lo.shape == (100,)

    def test_stricter_alpha_wider_intervals(self):
        model, X_s, y_s, X_t, _ = make_regression_problem()
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_s, X_t)

        cp90 = ShiftRobustConformal(model=model, adaptor=adaptor, alpha=0.10)
        cp50 = ShiftRobustConformal(model=model, adaptor=adaptor, alpha=0.50)

        # Same adaptor, need to refit separately
        a2 = CovariateShiftAdaptor(method="rulsif")
        a2.fit(X_s, X_t)
        cp50.adaptor = a2

        cp90.calibrate(X_s, y_s)
        cp50.calibrate(X_s, y_s)

        w90 = cp90.interval_width(X_t).mean()
        w50 = cp50.interval_width(X_t).mean()
        assert w90 >= w50


# -----------------------------------------------------------------------
# LR-QR conformal
# -----------------------------------------------------------------------

class TestLRQRConformal:
    def _setup(self, n_cal=400):
        model, X_s, y_s, X_t, y_t = make_regression_problem(n_source=n_cal)
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_s, X_t)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor, method="lrqr", alpha=0.1)
        cp.calibrate(X_s, y_s)
        return cp, X_t, y_t

    def test_calibrate_and_predict(self):
        cp, X_t, _ = self._setup()
        lo, hi = cp.predict_interval(X_t)
        assert lo.shape == (len(X_t),)
        assert hi.shape == (len(X_t),)

    def test_lower_lt_upper(self):
        cp, X_t, _ = self._setup()
        lo, hi = cp.predict_interval(X_t)
        assert (hi > lo).all()

    def test_lrqr_widths_may_vary(self):
        cp, X_t, _ = self._setup()
        widths = cp.interval_width(X_t)
        # LR-QR widths can vary across X (adaptive)
        # Just check they are all non-negative
        assert (widths >= 0).all()

    def test_coverage_reasonable(self):
        cp, X_t, y_t = self._setup()
        cov = cp.empirical_coverage(X_t, y_t)
        assert 0.6 <= cov <= 1.0

    def test_small_calibration_warns(self):
        model, X_s, y_s, X_t, _ = make_regression_problem(n_source=50)
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_s, X_t)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor, method="lrqr")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.calibrate(X_s, y_s)
            assert any("LR-QR" in str(warning.message) for warning in w)

    def test_lrqr_fitted_attribute(self):
        cp, _, _ = self._setup()
        assert cp._lrqr_threshold is not None
        assert cp._lrqr_threshold._fitted


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestConformalEdgeCases:
    def test_predict_1d_test_input(self):
        model, X_s, y_s, X_t, _ = make_regression_problem(d=1)
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_s, X_t)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor)
        cp.calibrate(X_s, y_s)
        lo, hi = cp.predict_interval(X_t[:, 0])  # Pass 1D
        assert lo.shape == (len(X_t),)

    def test_ridge_model(self):
        rng = np.random.default_rng(70)
        X = rng.normal(0, 1, (300, 4))
        y = X[:, 0] + rng.normal(0, 0.3, 300)
        model = Ridge().fit(X, y)
        X_t = rng.normal(0.5, 1, (100, 4))
        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X, X_t)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor)
        cp.calibrate(X, y)
        lo, hi = cp.predict_interval(X_t)
        assert (hi >= lo).all()
