"""Extended tests for CovariateShiftAdaptor — additional edge cases and parameter paths."""

import numpy as np
import pytest

from insurance_covariate_shift.adaptor import CovariateShiftAdaptor
from insurance_covariate_shift._types import CovariateShiftConfig
from insurance_covariate_shift.report import ShiftDiagnosticReport


RNG = np.random.default_rng(555)


def _books(n_s=300, n_t=200, d=4, shift=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n_s, d)), rng.normal(shift, 1.2, (n_t, d))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestAdaptorConstructorEdges:
    def test_clip_quantile_exactly_one_accepted(self):
        # 1.0 is the upper boundary and should be accepted
        a = CovariateShiftAdaptor(clip_quantile=1.0)
        assert a.clip_quantile == 1.0

    def test_clip_quantile_exactly_half_rejected(self):
        with pytest.raises(ValueError, match="clip_quantile"):
            CovariateShiftAdaptor(clip_quantile=0.5)

    def test_clip_quantile_above_half_accepted(self):
        a = CovariateShiftAdaptor(clip_quantile=0.51)
        assert a.clip_quantile == pytest.approx(0.51)

    def test_empty_categorical_cols_default(self):
        a = CovariateShiftAdaptor()
        assert a.categorical_cols == []

    def test_custom_config_stored(self):
        cfg = CovariateShiftConfig(ess_severe_threshold=0.2)
        a = CovariateShiftAdaptor(config=cfg)
        assert a.config.ess_severe_threshold == 0.2

    def test_default_config_created_when_none(self):
        a = CovariateShiftAdaptor(config=None)
        assert isinstance(a.config, CovariateShiftConfig)

    def test_catboost_params_stored(self):
        a = CovariateShiftAdaptor(catboost_iterations=50, catboost_depth=3)
        assert a.catboost_iterations == 50
        assert a.catboost_depth == 3

    def test_rulsif_alpha_stored(self):
        a = CovariateShiftAdaptor(rulsif_alpha=0.2)
        assert a.rulsif_alpha == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# RuLSIF adaptor extended
# ---------------------------------------------------------------------------

class TestRuLSIFAdaptorExtended:
    def test_clip_threshold_set_after_fit(self):
        X_s, X_t = _books(seed=10)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        assert np.isfinite(a.clip_threshold_)
        assert a.clip_threshold_ >= 0

    def test_weights_clipped_at_threshold(self):
        X_s, X_t = _books(seed=11, shift=2.0)
        a = CovariateShiftAdaptor(method="rulsif", clip_quantile=0.90)
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert w.max() <= a.clip_threshold_ + 1e-9

    def test_fit_transform_equivalent_to_fit_then_predict(self):
        X_s, X_t = _books(seed=12)
        a1 = CovariateShiftAdaptor(method="rulsif")
        w1 = a1.fit_transform(X_s, X_t)

        a2 = CovariateShiftAdaptor(method="rulsif")
        a2.fit(X_s, X_t)
        w2 = a2.importance_weights(X_s)

        np.testing.assert_allclose(w1, w2, atol=1e-10)

    def test_raw_weights_stored_after_fit(self):
        X_s, X_t = _books(seed=13)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        assert a._raw_weights_source is not None
        assert a._raw_weights_source.shape == (300,)

    def test_feature_names_auto_generated_when_not_provided(self):
        X_s, X_t = _books(seed=14)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        assert len(a._feature_names) == 4
        assert all(n.startswith("f") for n in a._feature_names)

    def test_feature_names_provided_stored(self):
        X_s, X_t = _books(seed=15)
        names = ["age", "ncb", "veh_age", "region"]
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t, feature_names=names)
        assert a._feature_names == names

    def test_1d_source_fit(self):
        rng = np.random.default_rng(16)
        X_s = rng.normal(0, 1, 200)
        X_t = rng.normal(0.5, 1, 150)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert w.shape == (200,)

    def test_importance_weights_1d_input_broadcast(self):
        rng = np.random.default_rng(17)
        X_s = rng.normal(0, 1, (200, 1))
        X_t = rng.normal(1, 1, (150, 1))
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        # Pass 1D array
        w = a.importance_weights(X_s[:, 0])
        assert w.shape == (200,)

    def test_shift_diagnostic_returns_report(self):
        X_s, X_t = _books(seed=18)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        report = a.shift_diagnostic()
        assert isinstance(report, ShiftDiagnosticReport)

    def test_shift_diagnostic_labels_passed_through(self):
        X_s, X_t = _books(seed=19)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        report = a.shift_diagnostic(source_label="Renewals", target_label="New Bizz")
        assert report.source_label == "Renewals"
        assert report.target_label == "New Bizz"

    def test_n_source_n_target_stored(self):
        X_s, X_t = _books(n_s=250, n_t=180, seed=20)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        assert a._n_source == 250
        assert a._n_target == 180

    def test_repr_string(self):
        a = CovariateShiftAdaptor(method="kliep")
        assert "kliep" in repr(a)

    def test_column_mismatch_raises(self):
        X_s = np.random.default_rng(21).normal(0, 1, (100, 4))
        X_t = np.random.default_rng(22).normal(0, 1, (80, 5))
        a = CovariateShiftAdaptor(method="rulsif")
        with pytest.raises(ValueError, match="columns"):
            a.fit(X_s, X_t)

    def test_importance_weights_before_fit_raises(self):
        a = CovariateShiftAdaptor(method="rulsif")
        with pytest.raises(RuntimeError, match="fit"):
            a.importance_weights(np.zeros((5, 4)))

    def test_shift_diagnostic_before_fit_raises(self):
        a = CovariateShiftAdaptor(method="rulsif")
        with pytest.raises(RuntimeError, match="fit"):
            a.shift_diagnostic()


# ---------------------------------------------------------------------------
# Exposure column handling
# ---------------------------------------------------------------------------

class TestExposureColumn:
    def _with_exposure(self, seed=30, n_s=200, n_t=150):
        rng = np.random.default_rng(seed)
        X_feat_s = rng.normal(0, 1, (n_s, 3))
        exp_s = rng.uniform(0.3, 1.0, (n_s, 1))
        X_s = np.hstack([X_feat_s, exp_s])

        X_feat_t = rng.normal(0.5, 1, (n_t, 3))
        exp_t = rng.uniform(0.3, 1.0, (n_t, 1))
        X_t = np.hstack([X_feat_t, exp_t])
        return X_s, X_t

    def test_exposure_col_excluded_from_feature_cols(self):
        X_s, X_t = self._with_exposure(seed=30)
        a = CovariateShiftAdaptor(method="rulsif", exposure_col=3)
        a.fit(X_s, X_t)
        assert 3 not in a._feature_cols

    def test_weights_with_exposure_non_negative(self):
        X_s, X_t = self._with_exposure(seed=31)
        a = CovariateShiftAdaptor(method="rulsif", exposure_col=3)
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert (w >= 0).all()

    def test_mean_exposure_stored(self):
        rng = np.random.default_rng(32)
        X_feat_s = rng.normal(0, 1, (200, 2))
        exp_s = np.full((200, 1), 0.75)
        X_s = np.hstack([X_feat_s, exp_s])
        X_feat_t = rng.normal(0, 1, (150, 2))
        exp_t = np.ones((150, 1))
        X_t = np.hstack([X_feat_t, exp_t])

        a = CovariateShiftAdaptor(method="rulsif", exposure_col=2)
        a.fit(X_s, X_t)
        assert a._mean_source_exposure == pytest.approx(0.75, rel=1e-6)

    def test_no_exposure_col_mean_is_one(self):
        X_s, X_t = _books(seed=33)
        a = CovariateShiftAdaptor(method="rulsif")
        a.fit(X_s, X_t)
        assert a._mean_source_exposure == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# KLIEP adaptor extended
# ---------------------------------------------------------------------------

class TestKLIEPAdaptorExtended:
    def test_weights_shape(self):
        X_s, X_t = _books(seed=40)
        a = CovariateShiftAdaptor(method="kliep")
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert w.shape == (300,)

    def test_weights_non_negative(self):
        X_s, X_t = _books(seed=41)
        a = CovariateShiftAdaptor(method="kliep")
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert (w >= 0).all()

    def test_feature_importances_none_for_kliep(self):
        X_s, X_t = _books(seed=42)
        a = CovariateShiftAdaptor(method="kliep")
        a.fit(X_s, X_t)
        # KLIEP does not set feature importances
        assert a._feature_importances is None

    def test_fit_transform_kliep(self):
        X_s, X_t = _books(seed=43)
        a = CovariateShiftAdaptor(method="kliep")
        w = a.fit_transform(X_s, X_t)
        assert w.shape[0] == 300


# ---------------------------------------------------------------------------
# CatBoost adaptor (skip if not installed)
# ---------------------------------------------------------------------------

class TestCatBoostAdaptorExtended:
    @pytest.fixture(autouse=True)
    def skip_if_no_catboost(self):
        pytest.importorskip("catboost")

    def test_feature_importances_length(self):
        X_s, X_t = _books(n_s=200, n_t=150, d=5, seed=50)
        a = CovariateShiftAdaptor(method="catboost", catboost_iterations=30)
        a.fit(X_s, X_t)
        assert len(a._feature_importances) == 5

    def test_weights_non_negative(self):
        X_s, X_t = _books(n_s=200, n_t=150, seed=51)
        a = CovariateShiftAdaptor(method="catboost", catboost_iterations=30)
        a.fit(X_s, X_t)
        w = a.importance_weights(X_s)
        assert (w >= 0).all()

    def test_fit_transform(self):
        X_s, X_t = _books(n_s=200, n_t=150, seed=52)
        a = CovariateShiftAdaptor(method="catboost", catboost_iterations=30)
        w = a.fit_transform(X_s, X_t)
        assert w.shape == (200,)

    def test_shift_diagnostic_has_importances(self):
        X_s, X_t = _books(n_s=200, n_t=150, seed=53)
        a = CovariateShiftAdaptor(method="catboost", catboost_iterations=30)
        a.fit(X_s, X_t)
        report = a.shift_diagnostic()
        fi = report.feature_importance()
        assert len(fi) == 4
        assert abs(sum(fi.values()) - 1.0) < 1e-6
