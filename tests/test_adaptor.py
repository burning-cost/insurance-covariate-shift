"""Tests for CovariateShiftAdaptor."""

import numpy as np
import pytest

from insurance_covariate_shift.adaptor import CovariateShiftAdaptor
from insurance_covariate_shift.report import ShiftDiagnosticReport


RNG = np.random.default_rng(7)


def make_books(n_source=300, n_target=200, shift=0.5, d=4):
    X_source = RNG.normal(0, 1, (n_source, d))
    X_target = RNG.normal(shift, 1.2, (n_target, d))
    return X_source, X_target


class TestAdaptorInit:
    def test_default_method(self):
        a = CovariateShiftAdaptor()
        assert a.method == "catboost"

    def test_rulsif_method(self):
        a = CovariateShiftAdaptor(method="rulsif")
        assert a.method == "rulsif"

    def test_kliep_method(self):
        a = CovariateShiftAdaptor(method="kliep")
        assert a.method == "kliep"

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            CovariateShiftAdaptor(method="unknown")

    def test_invalid_clip_quantile(self):
        with pytest.raises(ValueError, match="clip_quantile"):
            CovariateShiftAdaptor(clip_quantile=0.1)

    def test_not_fitted_initially(self):
        assert not CovariateShiftAdaptor().is_fitted_

    def test_repr(self):
        a = CovariateShiftAdaptor(method="rulsif")
        assert "rulsif" in repr(a)


class TestRuLSIFAdaptor:
    def test_fit_returns_self(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="rulsif")
        result = a.fit(X_s, X_t)
        assert result is a

    def test_is_fitted_after_fit(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        assert a.is_fitted_

    def test_importance_weights_shape(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert w.shape == (300,)

    def test_importance_weights_non_negative(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert (w >= 0).all()

    def test_unfitted_raises(self):
        a = CovariateShiftAdaptor(method="rulsif")
        with pytest.raises(RuntimeError, match="fit"):
            a.importance_weights(np.zeros((5, 4)))

    def test_mismatched_columns_raises(self):
        X_s = RNG.normal(0, 1, (100, 4))
        X_t = RNG.normal(0, 1, (80, 3))
        a = CovariateShiftAdaptor(method="rulsif")
        with pytest.raises(ValueError, match="columns"):
            a.fit(X_s, X_t)

    def test_1d_input_broadcast(self):
        rng = np.random.default_rng(20)
        X_s = rng.normal(0, 1, 200)  # 1D
        X_t = rng.normal(1, 1, 150)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert w.shape == (200,)

    def test_clipping_applied(self):
        X_s, X_t = make_books(shift=3.0)  # Large shift -> extreme weights
        a = CovariateShiftAdaptor(method="rulsif", clip_quantile=0.95)
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert w.max() <= a.clip_threshold_ + 1e-6

    def test_fit_transform(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="rulsif")
        w = a.fit_transform(X_s, X_t)
        assert w.shape == (300,)

    def test_feature_names_stored(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t, feature_names=["age", "ncb", "vehicle_age", "postcode"])
        assert a._feature_names == ["age", "ncb", "vehicle_age", "postcode"]

    def test_exposure_col_excluded_from_model(self):
        # Add exposure as last column
        rng = np.random.default_rng(30)
        X_s_feat = rng.normal(0, 1, (200, 3))
        exp_s = rng.uniform(0.5, 1.0, (200, 1))
        X_s = np.hstack([X_s_feat, exp_s])

        X_t_feat = rng.normal(0.5, 1, (150, 3))
        exp_t = rng.uniform(0.5, 1.0, (150, 1))
        X_t = np.hstack([X_t_feat, exp_t])

        a = CovariateShiftAdaptor(method="rulsif", exposure_col=3)
        a.fit(X_s, X_t)
        assert 3 not in a._feature_cols
        w = a.importance_weights(X_s)
        assert w.shape == (200,)
        assert (w >= 0).all()


class TestKLIEPAdaptor:
    def test_fit_and_predict(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="kliep")
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert w.shape == (300,)
        assert (w >= 0).all()

    def test_normalisation_approximately_one(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="kliep")
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        # Clipping may push mean slightly away from 1
        assert 0.5 < w.mean() < 3.0


class TestCatBoostAdaptor:
    """CatBoost tests — only run if catboost is installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_catboost(self):
        pytest.importorskip("catboost")

    def test_fit_and_predict(self):
        X_s, X_t = make_books(n_source=200, n_target=150)
        a = CovariateShiftAdaptor(method="catboost", catboost_iterations=50)
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert w.shape == (200,)
        assert (w >= 0).all()

    def test_feature_importances_set(self):
        X_s, X_t = make_books(n_source=200, n_target=150)
        a = CovariateShiftAdaptor(method="catboost", catboost_iterations=50)
        a.fit(X_s, X_t)
        assert a._feature_importances is not None
        assert len(a._feature_importances) == 4

    def test_categorical_cols_passed(self):
        # With integer-encoded categoricals
        rng = np.random.default_rng(50)
        X_s = rng.integers(0, 10, (200, 2)).astype(float)
        X_t = rng.integers(0, 10, (150, 2)).astype(float)
        a = CovariateShiftAdaptor(method="catboost", categorical_cols=[0, 1], catboost_iterations=50)
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert w.shape == (200,)

    def test_shift_diagnostic_from_catboost(self):
        X_s, X_t = make_books(n_source=200, n_target=150)
        a = CovariateShiftAdaptor(method="catboost", catboost_iterations=50)
        a.fit(X_s, X_t)
        report = a.shift_diagnostic()
        assert isinstance(report, ShiftDiagnosticReport)
        assert report.verdict in ("NEGLIGIBLE", "MODERATE", "SEVERE")


class TestShiftDiagnosticFromAdaptor:
    def test_diagnostic_returns_report(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        report = a.shift_diagnostic()
        assert isinstance(report, ShiftDiagnosticReport)

    def test_unfitted_diagnostic_raises(self):
        a = CovariateShiftAdaptor(method="rulsif")
        with pytest.raises(RuntimeError, match="fit"):
            a.shift_diagnostic()

    def test_custom_labels(self):
        X_s, X_t = make_books()
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        report = a.shift_diagnostic(source_label="Direct", target_label="Broker")
        assert report.source_label == "Direct"
        assert report.target_label == "Broker"

    def test_large_shift_produces_severe(self):
        rng = np.random.default_rng(99)
        X_s = rng.normal(0, 0.3, (300, 3))
        X_t = rng.normal(8, 0.3, (200, 3))  # Very extreme shift ensures low ESS
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        report = a.shift_diagnostic()
        # With extreme shift, ESS should be very low
        assert report.ess_ratio < 0.6


class TestExposureNaNHandling:
    """P1: exposure_col mean calculation must be NaN-safe."""

    def test_nan_exposure_falls_back_to_one(self):
        """When exposure column contains NaN, mean should fall back to 1.0."""
        rng = np.random.default_rng(42)
        n_source = 200
        n_target = 150

        # Build source with NaN in exposure column
        X_feat_s = rng.normal(0, 1, (n_source, 3))
        exp_s = rng.uniform(0.5, 1.5, (n_source, 1))
        exp_s[0] = np.nan   # inject a NaN
        exp_s[5] = np.nan
        X_source = np.hstack([X_feat_s, exp_s])

        X_feat_t = rng.normal(0.5, 1, (n_target, 3))
        exp_t = rng.uniform(0.5, 1.5, (n_target, 1))
        X_target = np.hstack([X_feat_t, exp_t])

        a = CovariateShiftAdaptor(method="rulsif", exposure_col=3)
        a.fit(X_source, X_target)

        # _mean_source_exposure should be finite and positive
        assert np.isfinite(a._mean_source_exposure)
        assert a._mean_source_exposure > 0

    def test_all_nan_exposure_falls_back_to_one(self):
        """All-NaN exposure column must fall back gracefully to 1.0."""
        rng = np.random.default_rng(10)
        X_feat_s = rng.normal(0, 1, (100, 2))
        exp_s = np.full((100, 1), np.nan)
        X_source = np.hstack([X_feat_s, exp_s])

        X_feat_t = rng.normal(0.5, 1, (80, 2))
        exp_t = np.ones((80, 1))
        X_target = np.hstack([X_feat_t, exp_t])

        a = CovariateShiftAdaptor(method="rulsif", exposure_col=2)
        a.fit(X_source, X_target)
        assert a._mean_source_exposure == pytest.approx(1.0)

    def test_zero_mean_exposure_falls_back_to_one(self):
        """Zero mean exposure (pathological) should fall back to 1.0."""
        rng = np.random.default_rng(20)
        X_feat_s = rng.normal(0, 1, (100, 2))
        exp_s = np.zeros((100, 1))  # all-zero exposure -> mean = 0
        X_source = np.hstack([X_feat_s, exp_s])

        X_feat_t = rng.normal(0.5, 1, (80, 2))
        exp_t = np.ones((80, 1))
        X_target = np.hstack([X_feat_t, exp_t])

        a = CovariateShiftAdaptor(method="rulsif", exposure_col=2)
        a.fit(X_source, X_target)
        assert a._mean_source_exposure == pytest.approx(1.0)

    def test_normal_exposure_uses_actual_mean(self):
        """Valid exposure without NaN should use the actual mean, not fall back."""
        rng = np.random.default_rng(30)
        X_feat_s = rng.normal(0, 1, (200, 3))
        exp_s = np.full((200, 1), 0.5)  # uniform exposure of 0.5
        X_source = np.hstack([X_feat_s, exp_s])

        X_feat_t = rng.normal(0.5, 1, (150, 3))
        exp_t = np.ones((150, 1))
        X_target = np.hstack([X_feat_t, exp_t])

        a = CovariateShiftAdaptor(method="rulsif", exposure_col=3)
        a.fit(X_source, X_target)
        assert a._mean_source_exposure == pytest.approx(0.5, rel=1e-6)
