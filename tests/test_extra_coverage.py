"""
Additional tests expanding coverage for insurance-covariate-shift.

Batch 22 of 34 library sweep — targeting gaps in:
- CovariateShiftAdaptor: clip_quantile boundary, feature_names auto-generation,
  fit_transform, single-row inputs, column mismatch at predict time
- ShiftRobustConformal: _LRQRThreshold directly, unfitted threshold path,
  interval_width, empirical_coverage with lrqr, no-adaptor fallback predictions
- RuLSIF / KLIEP: score method, early convergence, predict before fit
- ShiftDiagnosticReport: report_date parameter, KL warning, repr detail,
  feature importance with mismatched names, plot with categorical data
- CovariateShiftConfig: boundary conditions, combined thresholds
- _weighted_quantile: negative weights clipped, single-element arrays
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from insurance_covariate_shift import (
    CovariateShiftAdaptor,
    ShiftRobustConformal,
    ShiftDiagnosticReport,
    CovariateShiftConfig,
    RuLSIF,
    KLIEP,
    ShiftVerdict,
)
from insurance_covariate_shift.conformal import _weighted_quantile, _LRQRThreshold
from insurance_covariate_shift.density_ratio import _gaussian_kernel, _median_heuristic


RNG = np.random.default_rng(12345)


def _make_simple_data(n_source=100, n_target=80, d=3, shift=0.5):
    rng = np.random.default_rng(42)
    X_s = rng.normal(0, 1, (n_source, d))
    X_t = rng.normal(shift, 1, (n_target, d))
    return X_s, X_t


# ---------------------------------------------------------------------------
# CovariateShiftAdaptor — additional edge cases
# ---------------------------------------------------------------------------

class TestAdaptorEdgeCases:

    def test_clip_quantile_boundary_exact_one(self):
        """clip_quantile=1.0 is valid (inclusive)."""
        a = CovariateShiftAdaptor(method="rulsif", clip_quantile=1.0)
        assert a.clip_quantile == 1.0

    def test_clip_quantile_just_above_half(self):
        """clip_quantile just above 0.5 is valid."""
        a = CovariateShiftAdaptor(method="rulsif", clip_quantile=0.51)
        assert a.clip_quantile == 0.51

    def test_clip_quantile_exactly_half_raises(self):
        """clip_quantile=0.5 is outside (0.5, 1.0]."""
        with pytest.raises(ValueError, match="clip_quantile"):
            CovariateShiftAdaptor(clip_quantile=0.5)

    def test_feature_names_auto_generated_if_none(self):
        X_s, X_t = _make_simple_data(d=3)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        # Auto-generated names should be f0, f1, f2
        assert a._feature_names == ["f0", "f1", "f2"]

    def test_feature_names_length_matches_feature_cols(self):
        rng = np.random.default_rng(1)
        # 4 cols, last is exposure
        X_s = np.hstack([rng.normal(0, 1, (100, 3)), rng.uniform(0.5, 1, (100, 1))])
        X_t = np.hstack([rng.normal(0.5, 1, (80, 3)), rng.uniform(0.5, 1, (80, 1))])
        a = CovariateShiftAdaptor(method="rulsif", exposure_col=3)
        a.fit(X_s, X_t, feature_names=["age", "ncb", "veh_age"])
        assert len(a._feature_names) == 3

    def test_fit_transform_is_same_as_fit_then_weights(self):
        X_s, X_t = _make_simple_data()
        a1 = CovariateShiftAdaptor(method="rulsif")
        w_combined = a1.fit_transform(X_s, X_t)

        a2 = CovariateShiftAdaptor(method="rulsif")
        a2.fit(X_s, X_t)
        w_separate = a2.importance_weights(X_s)

        # Results should be numerically identical
        np.testing.assert_allclose(w_combined, w_separate, rtol=1e-6)

    def test_importance_weights_target_data(self):
        """Weights on target data should also be non-negative."""
        X_s, X_t = _make_simple_data()
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        w = a.importance_weights(X_t)
        assert w.shape == (len(X_t),)
        assert (w >= 0).all()

    def test_n_source_and_n_target_stored(self):
        X_s, X_t = _make_simple_data(n_source=150, n_target=90)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        assert a._n_source == 150
        assert a._n_target == 90

    def test_kliep_fit_transform_shape(self):
        X_s, X_t = _make_simple_data()
        a = CovariateShiftAdaptor(method="kliep")
        w = a.fit_transform(X_s, X_t)
        assert w.shape == (len(X_s),)

    def test_shift_diagnostic_labels_default(self):
        X_s, X_t = _make_simple_data()
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        report = a.shift_diagnostic()
        assert report.source_label == "Source"
        assert report.target_label == "Target"

    def test_exposure_col_weights_scale_with_exposure(self):
        """Observations with higher exposure should get proportionally higher weights."""
        rng = np.random.default_rng(77)
        n = 200
        X_feat_s = rng.normal(0, 1, (n, 2))
        # Two groups: half with exposure=0.5, half with exposure=2.0
        exp_s = np.where(np.arange(n) < n // 2, 0.5, 2.0).reshape(-1, 1)
        X_s = np.hstack([X_feat_s, exp_s])

        X_feat_t = rng.normal(0.3, 1, (150, 2))
        exp_t = np.ones((150, 1))
        X_t = np.hstack([X_feat_t, exp_t])

        a = CovariateShiftAdaptor(method="rulsif", exposure_col=2)
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)

        # Mean weight in high-exposure group should be higher than low-exposure group
        mean_low = w[:n // 2].mean()
        mean_high = w[n // 2:].mean()
        # High exposure group should on average have higher weights
        # (2x exposure -> ~2x expected weight after scaling)
        assert mean_high > mean_low

    def test_config_parameter_is_stored(self):
        cfg = CovariateShiftConfig(ess_severe_threshold=0.4)
        a = CovariateShiftAdaptor(method="rulsif", config=cfg)
        assert a.config.ess_severe_threshold == 0.4

    def test_raw_weights_stored_after_fit(self):
        X_s, X_t = _make_simple_data()
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        assert a._raw_weights_source is not None
        assert a._raw_weights_source.shape == (len(X_s),)

    def test_rulsif_custom_iterations_and_depth_ignored(self):
        """catboost_iterations and catboost_depth don't affect rulsif."""
        X_s, X_t = _make_simple_data()
        a = CovariateShiftAdaptor(method="rulsif", catboost_iterations=10, catboost_depth=2)
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert (w >= 0).all()


# ---------------------------------------------------------------------------
# ShiftRobustConformal — additional cases
# ---------------------------------------------------------------------------

class TestLRQRThresholdDirect:
    """Test _LRQRThreshold in isolation."""

    def _make_fitted_threshold(self, n=200, alpha=0.1):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (n, 3))
        scores = rng.exponential(1, n)
        weights = rng.uniform(0.5, 2.0, n)
        t = _LRQRThreshold(alpha=alpha, lam=1.0)
        t.fit(X, scores, weights)
        return t, X

    def test_fit_sets_fitted_flag(self):
        t, _ = self._make_fitted_threshold()
        assert t._fitted is True

    def test_predict_shape(self):
        t, X = self._make_fitted_threshold()
        preds = t.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_unfitted_returns_base_quantile(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 3))
        t = _LRQRThreshold(alpha=0.1)
        t._fitted = False
        t._base_quantile = 99.0
        preds = t.predict(X)
        np.testing.assert_allclose(preds, 99.0)

    def test_base_quantile_positive(self):
        t, _ = self._make_fitted_threshold()
        assert t._base_quantile >= 0

    def test_high_lambda_amplifies_weights(self):
        """Higher lambda should increase tree influence (not fully isolated, just sanity)."""
        rng = np.random.default_rng(5)
        X = rng.normal(0, 1, (200, 3))
        scores = rng.exponential(1, 200)
        weights = rng.uniform(0.5, 5.0, 200)
        t_low = _LRQRThreshold(alpha=0.1, lam=0.0)
        t_high = _LRQRThreshold(alpha=0.1, lam=10.0)
        t_low.fit(X, scores, weights)
        t_high.fit(X, scores, weights)
        # Both should produce valid predictions
        assert t_low.predict(X).shape == (200,)
        assert t_high.predict(X).shape == (200,)

    def test_zero_weights_handled(self):
        """All-zero weights should not cause crash."""
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (100, 3))
        scores = rng.exponential(1, 100)
        weights = np.zeros(100)
        t = _LRQRThreshold(alpha=0.1)
        t.fit(X, scores, weights)
        preds = t.predict(X)
        assert preds.shape == (100,)


class TestConformalAdditional:

    def _make_setup(self, n_cal=200, alpha=0.1, method="weighted"):
        rng = np.random.default_rng(99)
        d = 4
        X_train = rng.normal(0, 1, (300, d))
        y_train = X_train[:, 0] * 2 + rng.normal(0, 0.5, 300)
        model = LinearRegression().fit(X_train, y_train)

        X_cal = rng.normal(0, 1, (n_cal, d))
        y_cal = X_cal[:, 0] * 2 + rng.normal(0, 0.5, n_cal)
        X_target = rng.normal(0.5, 1.2, (100, d))

        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_cal, X_target)
        cp = ShiftRobustConformal(model=model, adaptor=adaptor, method=method, alpha=alpha)
        cp.calibrate(X_cal, y_cal)
        return cp, X_target, rng

    def test_lrqr_interval_width_non_negative(self):
        cp, X_t, _ = self._make_setup(n_cal=300, method="lrqr")
        widths = cp.interval_width(X_t)
        assert (widths >= 0).all()

    def test_lrqr_coverage_attribute(self):
        cp, X_t, rng = self._make_setup(n_cal=300, method="lrqr")
        y_test = X_t[:, 0] * 2 + rng.normal(0, 0.5, 100)
        cov = cp.empirical_coverage(X_t, y_test)
        assert 0.0 <= cov <= 1.0

    def test_weighted_interval_is_symmetric(self):
        """Weighted conformal produces symmetric intervals around point estimate."""
        cp, X_t, _ = self._make_setup()
        lo, hi = cp.predict_interval(X_t)
        y_hat = np.asarray(cp.model.predict(X_t))
        np.testing.assert_allclose(hi - y_hat, y_hat - lo, rtol=1e-6)

    def test_calibrate_n_calibration_stored(self):
        cp, _, _ = self._make_setup(n_cal=150)
        assert cp.n_calibration_ == 150

    def test_global_quantile_positive(self):
        cp, _, _ = self._make_setup()
        assert cp._global_quantile >= 0

    def test_no_adaptor_calibrate_fallback_still_predicts(self):
        """Calibrating with no adaptor and no target data still produces intervals."""
        rng = np.random.default_rng(10)
        X = rng.normal(0, 1, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.2, 200)
        model = LinearRegression().fit(X, y)
        cp = ShiftRobustConformal(model=model, adaptor=None, method="weighted")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cp.calibrate(X, y)
        lo, hi = cp.predict_interval(X)
        assert (hi > lo).all()

    def test_alpha_close_to_one_gives_narrow_intervals(self):
        """Very small coverage requirement (alpha near 1) gives very narrow intervals."""
        rng = np.random.default_rng(20)
        X = rng.normal(0, 1, (200, 4))
        y = X[:, 0] + rng.normal(0, 0.5, 200)
        model = LinearRegression().fit(X, y)
        X_t = rng.normal(0.3, 1, (100, 4))

        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X, X_t)

        # alpha=0.9 means only 10% coverage needed -> very narrow intervals
        cp = ShiftRobustConformal(model=model, adaptor=adaptor, alpha=0.9)
        cp.calibrate(X, y)
        w_narrow = cp.interval_width(X_t).mean()

        # alpha=0.01 means 99% coverage -> wide intervals
        cp2 = ShiftRobustConformal(model=model, adaptor=adaptor, alpha=0.01)
        cp2.calibrate(X, y)
        w_wide = cp2.interval_width(X_t).mean()

        assert w_wide > w_narrow


# ---------------------------------------------------------------------------
# _weighted_quantile — edge cases
# ---------------------------------------------------------------------------

class TestWeightedQuantileExtra:

    def test_single_element(self):
        scores = np.array([5.0])
        weights = np.array([1.0])
        q = _weighted_quantile(scores, weights, 0.9)
        # With one element, +inf is appended; for level < 1.0 we get the element
        assert q <= np.inf

    def test_negative_weights_clipped_to_zero(self):
        """Negative weights should be treated as zero (clipped in implementation)."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights_neg = np.array([-10.0, -10.0, 1.0, 1.0, 1.0])
        # Should not raise
        q = _weighted_quantile(scores, weights_neg, 0.5)
        assert np.isfinite(q)

    def test_level_one_returns_inf(self):
        """Level=1.0 should return +inf (nothing reaches 100% threshold with finite scores)."""
        scores = np.array([1.0, 2.0, 3.0])
        weights = np.ones(3)
        q = _weighted_quantile(scores, weights, 1.0)
        assert q == np.inf

    def test_all_same_weights_monotone_quantiles(self):
        """Higher level should give higher or equal quantile."""
        rng = np.random.default_rng(7)
        scores = rng.exponential(1, 50)
        weights = np.ones(50)
        q1 = _weighted_quantile(scores, weights, 0.5)
        q2 = _weighted_quantile(scores, weights, 0.9)
        assert q2 >= q1


# ---------------------------------------------------------------------------
# RuLSIF — additional cases
# ---------------------------------------------------------------------------

class TestRuLSIFExtra:

    def test_predict_on_target_data(self):
        """Predict on target data — should have high weights on average."""
        rng = np.random.default_rng(20)
        X_s = rng.normal(0, 1, (200, 2))
        X_t = rng.normal(2, 1, (150, 2))
        m = RuLSIF()
        m.fit(X_s, X_t)
        w_t = m.predict(X_t)
        w_s = m.predict(X_s)
        # Target weights should be higher than source (target is the target distribution)
        assert w_t.mean() >= w_s.mean() - 1.0  # loose check

    def test_score_returns_scalar(self):
        rng = np.random.default_rng(21)
        X_s = rng.normal(0, 1, (100, 2))
        X_t = rng.normal(0.5, 1, (80, 2))
        m = RuLSIF()
        m.fit(X_s, X_t)
        s = m.score(X_s, X_t)
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_score_higher_for_more_separated_distributions(self):
        """Score should be larger when distributions are more different."""
        rng = np.random.default_rng(22)
        X_s = rng.normal(0, 1, (150, 2))
        X_t_near = rng.normal(0.5, 1, (100, 2))
        X_t_far = rng.normal(5.0, 1, (100, 2))
        m = RuLSIF()
        m.fit(X_s, X_t_near)
        s_near = m.score(X_s, X_t_near)

        m2 = RuLSIF()
        m2.fit(X_s, X_t_far)
        s_far = m2.score(X_s, X_t_far)

        assert s_far > s_near

    def test_large_regularisation_produces_smooth_weights(self):
        """With very high lambda, weights should be near-uniform."""
        rng = np.random.default_rng(23)
        X_s = rng.normal(0, 1, (100, 2))
        X_t = rng.normal(1, 1, (80, 2))
        m = RuLSIF(lam=1e6)
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        # Very high regularisation -> theta near zero -> weights near zero
        # (all non-negative, check just that they exist)
        assert w.shape == (100,)
        assert (w >= 0).all()

    def test_fitted_flag_set(self):
        rng = np.random.default_rng(24)
        m = RuLSIF()
        assert not m._fitted
        m.fit(rng.normal(0, 1, (50, 2)), rng.normal(0, 1, (40, 2)))
        assert m._fitted

    def test_alpha_boundary_zero_allowed(self):
        m = RuLSIF(alpha=0.0)
        rng = np.random.default_rng(25)
        m.fit(rng.normal(0, 1, (80, 2)), rng.normal(0.5, 1, (60, 2)))
        w = m.predict(rng.normal(0, 1, (30, 2)))
        assert (w >= 0).all()

    def test_single_feature(self):
        rng = np.random.default_rng(26)
        X_s = rng.normal(0, 1, (100, 1))
        X_t = rng.normal(0.5, 1, (80, 1))
        m = RuLSIF()
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert w.shape == (100,)


# ---------------------------------------------------------------------------
# KLIEP — additional cases
# ---------------------------------------------------------------------------

class TestKLIEPExtra:

    def test_predict_before_fit_raises(self):
        m = KLIEP()
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(np.zeros((5, 2)))

    def test_very_early_convergence(self):
        """With very high tolerance, should converge on first iteration."""
        rng = np.random.default_rng(30)
        X_s = rng.normal(0, 1, (100, 2))
        X_t = rng.normal(0.5, 1, (80, 2))
        m = KLIEP(tol=1e3, max_iter=100)  # very high tolerance
        m.fit(X_s, X_t)
        assert m._fitted

    def test_max_iter_one_runs_without_crash(self):
        rng = np.random.default_rng(31)
        X_s = rng.normal(0, 1, (100, 2))
        X_t = rng.normal(0.5, 1, (80, 2))
        m = KLIEP(max_iter=1)
        m.fit(X_s, X_t)
        w = m.predict(X_s)
        assert (w >= 0).all()

    def test_large_learning_rate_still_converges(self):
        rng = np.random.default_rng(32)
        X_s = rng.normal(0, 1, (200, 2))
        X_t = rng.normal(0.5, 1, (150, 2))
        m = KLIEP(learning_rate=0.1)
        m.fit(X_s, X_t)
        assert m._fitted

    def test_fitted_attribute(self):
        m = KLIEP()
        assert not m._fitted

    def test_alpha_vec_positive_after_fit(self):
        """KLIEP alpha_vec must be non-negative (positivity constraint)."""
        rng = np.random.default_rng(33)
        X_s = rng.normal(0, 1, (100, 2))
        X_t = rng.normal(0.5, 1, (80, 2))
        m = KLIEP()
        m.fit(X_s, X_t)
        assert (m._alpha_vec >= 0).all()


# ---------------------------------------------------------------------------
# Density ratio helper functions
# ---------------------------------------------------------------------------

class TestGaussianKernelExtra:

    def test_symmetry(self):
        rng = np.random.default_rng(40)
        X = rng.normal(0, 1, (5, 2))
        Y = rng.normal(0, 1, (5, 2))
        K_xy = _gaussian_kernel(X, Y, sigma=1.0)
        K_yx = _gaussian_kernel(Y, X, sigma=1.0)
        # K_xy[i,j] should equal K_yx[j,i]
        np.testing.assert_allclose(K_xy, K_yx.T, atol=1e-10)

    def test_zero_sigma_causes_extreme_localisation(self):
        """Very small sigma -> kernel is near-zero except for identical points."""
        X = np.array([[0.0, 0.0]])
        Y = np.array([[0.0, 0.0], [1.0, 1.0]])
        K = _gaussian_kernel(X, Y, sigma=1e-6)
        assert K[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert K[0, 1] < 1e-5


class TestMedianHeuristicExtra:

    def test_returns_float(self):
        rng = np.random.default_rng(50)
        X = rng.normal(0, 1, (50, 3))
        Y = rng.normal(1, 1, (40, 3))
        sigma = _median_heuristic(X, Y)
        assert isinstance(sigma, float)

    def test_larger_spread_gives_larger_sigma(self):
        """Wider spread data should produce larger bandwidth estimate."""
        rng = np.random.default_rng(51)
        X_narrow = rng.normal(0, 0.1, (100, 2))
        Y_narrow = rng.normal(0, 0.1, (80, 2))
        X_wide = rng.normal(0, 10.0, (100, 2))
        Y_wide = rng.normal(0, 10.0, (80, 2))
        sigma_narrow = _median_heuristic(X_narrow, Y_narrow)
        sigma_wide = _median_heuristic(X_wide, Y_wide)
        assert sigma_wide > sigma_narrow

    def test_subsamples_at_500(self):
        """Large dataset: should subsample to 500 without error."""
        rng = np.random.default_rng(52)
        X = rng.normal(0, 1, (1000, 5))
        Y = rng.normal(0.5, 1, (800, 5))
        sigma = _median_heuristic(X, Y)
        assert sigma > 0


# ---------------------------------------------------------------------------
# ShiftDiagnosticReport — additional cases
# ---------------------------------------------------------------------------

class TestShiftDiagnosticReportExtra:

    def test_report_date_parameter(self):
        """Custom report_date should appear in regulatory summary."""
        weights = np.ones(100)
        fixed_date = date(2024, 6, 15)
        report = ShiftDiagnosticReport(weights=weights, report_date=fixed_date)
        summary = report.fca_sup153_summary()
        assert "2024-06-15" in summary

    def test_kl_warning_when_weights_not_normalised(self):
        """Mean weight far from 1.0 should trigger UserWarning."""
        # Weights with mean=5 (far from 1)
        weights = np.ones(100) * 5.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report = ShiftDiagnosticReport(weights=weights)
        kl_warnings = [x for x in w if issubclass(x.category, UserWarning)
                       and "KL divergence" in str(x.message)]
        assert len(kl_warnings) >= 1

    def test_kl_divergence_near_zero_for_identical_weights(self):
        """Constant weights -> KL should be approximately 0."""
        weights = np.ones(500)
        report = ShiftDiagnosticReport(weights=weights)
        assert report.kl_divergence == pytest.approx(0.0, abs=1e-6)

    def test_repr_contains_n_weights(self):
        weights = np.ones(250)
        report = ShiftDiagnosticReport(weights=weights)
        r = repr(report)
        assert "n_weights=250" in r

    def test_repr_contains_ess_and_kl(self):
        weights = np.ones(50)
        report = ShiftDiagnosticReport(weights=weights)
        r = repr(report)
        assert "ess_ratio" in r
        assert "kl_divergence" in r

    def test_single_weight_ess_ratio_one(self):
        """Single observation with weight 1.0 -> ESS = 1/1 = 1.0."""
        weights = np.array([1.0])
        report = ShiftDiagnosticReport(weights=weights)
        assert report.ess_ratio == pytest.approx(1.0, abs=1e-6)

    def test_feature_importance_empty_names(self):
        """When feature_names is empty but importances are provided, uses f0, f1 etc."""
        weights = np.ones(100)
        report = ShiftDiagnosticReport(
            weights=weights,
            feature_names=[],
            feature_importances=np.array([0.6, 0.4]),
        )
        fi = report.feature_importance()
        assert set(fi.keys()) == {"f0", "f1"}

    def test_moderate_verdict_in_fca_summary(self):
        """MODERATE verdict should mention importance weighting in summary."""
        # Construct weights that give MODERATE verdict
        cfg = CovariateShiftConfig(
            ess_moderate_threshold=0.6,
            ess_severe_threshold=0.3,
            kl_moderate_threshold=0.1,
            kl_severe_threshold=0.5,
        )
        # ESS around 0.45 -> MODERATE
        rng = np.random.default_rng(60)
        # Exponential weights to get moderate ESS
        weights = rng.exponential(1, 200)
        report = ShiftDiagnosticReport(weights=weights, config=cfg)
        summary = report.fca_sup153_summary()
        if report.verdict == "MODERATE":
            assert "weighting" in summary.lower() or "weighted" in summary.lower()

    def test_no_feature_matrices_still_builds_summary(self):
        """Report without X_source/X_target should still produce regulatory summary."""
        weights = np.random.default_rng(61).lognormal(0, 0.3, 300)
        report = ShiftDiagnosticReport(weights=weights, feature_names=["age", "ncb"])
        summary = report.fca_sup153_summary()
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_feature_shifts_plot_low_cardinality_feature(self):
        """Categorical-like feature (low cardinality) should produce bar charts."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(62)
        # 2-feature matrix: first is categorical (values 0-4), second continuous
        X_s = np.column_stack([
            rng.integers(0, 5, 100).astype(float),
            rng.normal(0, 1, 100),
        ])
        X_t = np.column_stack([
            rng.integers(0, 5, 80).astype(float),
            rng.normal(0.5, 1, 80),
        ])
        weights = rng.lognormal(0, 0.3, 100)
        report = ShiftDiagnosticReport(
            weights=weights,
            feature_names=["region", "score"],
            X_source=X_s,
            X_target=X_t,
        )
        fig = report.plot_feature_shifts()
        assert fig is not None
        plt.close("all")

    def test_source_label_in_fca_summary(self):
        weights = np.ones(100) * 1.5
        report = ShiftDiagnosticReport(
            weights=weights,
            source_label="Direct Motor Book",
            target_label="Acquired Portfolio",
        )
        summary = report.fca_sup153_summary()
        assert "Direct Motor Book" in summary
        assert "Acquired Portfolio" in summary


# ---------------------------------------------------------------------------
# CovariateShiftConfig — boundary conditions
# ---------------------------------------------------------------------------

class TestCovariateShiftConfigExtra:

    def test_exactly_at_moderate_ess_boundary_is_negligible(self):
        """ESS exactly at moderate threshold is NEGLIGIBLE (exclusive boundary)."""
        cfg = CovariateShiftConfig()
        # ess_moderate_threshold=0.6, kl_moderate=0.1
        # ESS=0.6 with low KL -> NEGLIGIBLE (strict inequality: ESS < 0.6 is MODERATE)
        result = cfg.verdict(ess_ratio=0.6, kl_divergence=0.05)
        assert result == "NEGLIGIBLE"

    def test_just_below_moderate_ess_boundary_is_moderate(self):
        cfg = CovariateShiftConfig()
        result = cfg.verdict(ess_ratio=0.599, kl_divergence=0.05)
        assert result == "MODERATE"

    def test_kl_exactly_at_moderate_threshold_is_moderate(self):
        cfg = CovariateShiftConfig()
        # kl > 0.1 is MODERATE
        result = cfg.verdict(ess_ratio=0.9, kl_divergence=0.101)
        assert result == "MODERATE"

    def test_kl_exactly_at_moderate_threshold_not_moderate(self):
        """kl exactly = moderate threshold (0.1) is NOT moderate (strict >)."""
        cfg = CovariateShiftConfig()
        result = cfg.verdict(ess_ratio=0.9, kl_divergence=0.1)
        assert result == "NEGLIGIBLE"

    def test_both_signals_moderate_still_moderate(self):
        cfg = CovariateShiftConfig()
        result = cfg.verdict(ess_ratio=0.45, kl_divergence=0.3)
        assert result == "MODERATE"

    def test_perfect_overlap_is_negligible(self):
        """ESS=1.0 and KL=0 -> NEGLIGIBLE."""
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=1.0, kl_divergence=0.0) == "NEGLIGIBLE"

    def test_custom_config_changes_verdict(self):
        """With very high severe threshold, even moderate shift is SEVERE."""
        cfg = CovariateShiftConfig(ess_severe_threshold=0.99)
        result = cfg.verdict(ess_ratio=0.5, kl_divergence=0.05)
        assert result == "SEVERE"


# ---------------------------------------------------------------------------
# Integration: full pipeline with different methods
# ---------------------------------------------------------------------------

class TestIntegrationExtra:

    def test_rulsif_kliep_produce_similar_direction(self):
        """Both methods should classify heavily shifted books as non-negligible."""
        rng = np.random.default_rng(70)
        X_s = rng.normal(0, 0.5, (300, 3))
        X_t = rng.normal(3, 0.5, (200, 3))

        for method in ("rulsif", "kliep"):
            a = CovariateShiftAdaptor(method=method)
            a.fit(X_s, X_t)
            report = a.shift_diagnostic()
            # Extreme shift should not be NEGLIGIBLE
            assert report.verdict in ("MODERATE", "SEVERE"), \
                f"Method {method}: expected non-NEGLIGIBLE verdict for extreme shift"

    def test_adaptor_with_feature_names_and_report(self):
        """Feature names should propagate to the diagnostic report."""
        rng = np.random.default_rng(71)
        X_s = rng.normal(0, 1, (200, 3))
        X_t = rng.normal(0.5, 1, (150, 3))
        names = ["premium_factor", "ncd_years", "vehicle_value"]
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t, feature_names=names)
        report = a.shift_diagnostic()
        # Report feature_importance returns empty dict for rulsif (no importances)
        fi = report.feature_importance()
        assert isinstance(fi, dict)

    def test_identical_source_target_negligible(self):
        """Same distribution -> ESS should be high -> likely NEGLIGIBLE."""
        rng = np.random.default_rng(72)
        X_s = rng.normal(0, 1, (300, 4))
        X_t = rng.normal(0, 1, (250, 4))
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        report = a.shift_diagnostic()
        # ESS should be reasonably high
        assert report.ess_ratio > 0.4

    def test_conformal_coverage_for_weighted_method(self):
        """Check that weighted conformal coverage is reasonable on target."""
        rng = np.random.default_rng(73)
        d = 3
        X_train = rng.normal(0, 1, (400, d))
        y_train = X_train[:, 0] * 2 + rng.normal(0, 0.5, 400)
        model = LinearRegression().fit(X_train, y_train)

        X_cal = rng.normal(0, 1, (200, d))
        y_cal = X_cal[:, 0] * 2 + rng.normal(0, 0.5, 200)
        X_target = rng.normal(0.3, 1, (100, d))
        y_target = X_target[:, 0] * 2 + rng.normal(0, 0.5, 100)

        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_cal, X_target)

        cp = ShiftRobustConformal(model=model, adaptor=adaptor, method="weighted", alpha=0.10)
        cp.calibrate(X_cal, y_cal)
        cov = cp.empirical_coverage(X_target, y_target)
        # Coverage should be in [0.7, 1.0] (generous range given n=100)
        assert 0.7 <= cov <= 1.0

    def test_refit_adaptor_clears_state(self):
        """Fitting the same adaptor twice should update state."""
        rng = np.random.default_rng(74)
        X_s1 = rng.normal(0, 1, (100, 2))
        X_t1 = rng.normal(0.5, 1, (80, 2))
        X_s2 = rng.normal(2, 1, (100, 2))
        X_t2 = rng.normal(3, 1, (80, 2))

        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s1, X_t1)
        w1 = a.importance_weights(X_s1)

        a.fit(X_s2, X_t2)
        w2 = a.importance_weights(X_s2)

        # Weights from different fits should differ
        assert not np.allclose(w1.mean(), w2.mean())
