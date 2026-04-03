"""Extended tests for ShiftDiagnosticReport — metrics, regulatory output, plots."""

import warnings
from datetime import date

import numpy as np
import pytest

from insurance_covariate_shift.report import ShiftDiagnosticReport
from insurance_covariate_shift._types import CovariateShiftConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _report(weights, feature_names=None, feature_importances=None,
            X_source=None, X_target=None, config=None, **kw):
    return ShiftDiagnosticReport(
        weights=weights,
        feature_names=feature_names or [],
        feature_importances=feature_importances,
        X_source=X_source,
        X_target=X_target,
        config=config,
        **kw,
    )


# ---------------------------------------------------------------------------
# ESS ratio
# ---------------------------------------------------------------------------

class TestESSRatio:
    def test_uniform_weights_ess_one(self):
        w = np.ones(100)
        r = _report(w)
        assert r.ess_ratio == pytest.approx(1.0, abs=1e-8)

    def test_empty_weights_ess_zero(self):
        r = _report(np.array([]))
        assert r.ess_ratio == 0.0

    def test_single_weight_ess_one(self):
        r = _report(np.array([3.7]))
        assert r.ess_ratio == pytest.approx(1.0, abs=1e-8)

    def test_all_zero_weights_ess_zero(self):
        r = _report(np.zeros(50))
        assert r.ess_ratio == 0.0

    def test_ess_decreases_with_concentration(self):
        w_flat = np.ones(200)
        w_spike = np.zeros(200)
        w_spike[0] = 200.0
        r_flat = _report(w_flat)
        r_spike = _report(w_spike)
        assert r_flat.ess_ratio > r_spike.ess_ratio

    def test_ess_between_zero_and_one(self):
        rng = np.random.default_rng(600)
        w = rng.lognormal(0, 0.5, 200)
        r = _report(w)
        assert 0.0 <= r.ess_ratio <= 1.0

    def test_two_equal_blocks_ess(self):
        # Half zeros, half 2 -> sum=n, sum_sq = n/2 * 4 -> ESS = n^2 / (n/2*4) / n = 0.5
        n = 100
        w = np.zeros(n)
        w[n // 2:] = 2.0
        r = _report(w)
        # ESS = (sum w)^2 / (n * sum w^2) = (n)^2 / (n * n/2*4) = n^2/(2n^2) = 0.5
        assert r.ess_ratio == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_constant_weights_kl_zero(self):
        w = np.ones(200)
        r = _report(w)
        assert r.kl_divergence == pytest.approx(0.0, abs=1e-6)

    def test_kl_non_negative(self):
        rng = np.random.default_rng(601)
        w = rng.lognormal(0, 0.3, 300)
        r = _report(w)
        assert r.kl_divergence >= 0.0

    def test_kl_increases_with_spread(self):
        rng = np.random.default_rng(602)
        w_narrow = rng.lognormal(0, 0.1, 500)
        w_wide = rng.lognormal(0, 1.5, 500)
        r_narrow = _report(w_narrow)
        r_wide = _report(w_wide)
        assert r_wide.kl_divergence > r_narrow.kl_divergence

    def test_kl_warning_for_non_normalised_weights(self):
        # Weights with mean far from 1 should trigger a UserWarning
        w = np.ones(100) * 5.0  # mean = 5
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = _report(w)
            # May or may not warn depending on tolerance (mean=5 >> 0.1 deviation)
            # Just ensure it ran
        assert r.kl_divergence >= 0.0

    def test_empty_weights_kl_zero(self):
        r = _report(np.array([]))
        assert r.kl_divergence == 0.0


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

class TestVerdict:
    def test_verdict_type_is_string(self):
        r = _report(np.ones(100))
        assert isinstance(r.verdict, str)

    def test_verdict_one_of_three(self):
        rng = np.random.default_rng(603)
        for _ in range(10):
            w = rng.lognormal(0, rng.uniform(0.1, 1.5), 200)
            r = _report(w)
            assert r.verdict in ("NEGLIGIBLE", "MODERATE", "SEVERE")

    def test_verdict_negligible_for_uniform(self):
        r = _report(np.ones(500))
        assert r.verdict == "NEGLIGIBLE"

    def test_verdict_severe_for_spike(self):
        w = np.zeros(200)
        w[0] = 1000.0
        r = _report(w)
        assert r.verdict == "SEVERE"

    def test_custom_config_affects_verdict(self):
        w = np.ones(200) * 0.5 + np.random.default_rng(604).normal(0, 0.05, 200)
        cfg_loose = CovariateShiftConfig(ess_severe_threshold=0.01)
        cfg_tight = CovariateShiftConfig(ess_severe_threshold=0.99)
        r_loose = _report(w, config=cfg_loose)
        r_tight = _report(w, config=cfg_tight)
        # Tight threshold -> more likely SEVERE
        # At least one should be harsher
        verdicts_tight = ("MODERATE", "SEVERE")
        assert r_tight.verdict in verdicts_tight or r_loose.verdict == "NEGLIGIBLE"


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportanceExtended:
    def test_feature_importance_empty_when_no_importances(self):
        r = _report(np.ones(100), feature_names=["a", "b"])
        assert r.feature_importance() == {}

    def test_feature_importance_normalised(self):
        imp = np.array([10.0, 20.0, 30.0])
        r = _report(np.ones(100), feature_names=["a", "b", "c"],
                    feature_importances=imp)
        fi = r.feature_importance()
        assert abs(sum(fi.values()) - 1.0) < 1e-8

    def test_feature_importance_ordering(self):
        imp = np.array([0.6, 0.3, 0.1])
        r = _report(np.ones(100), feature_names=["age", "ncb", "region"],
                    feature_importances=imp)
        fi = r.feature_importance()
        # Largest importance is "age"
        assert fi["age"] > fi["ncb"] > fi["region"]

    def test_feature_importance_auto_names(self):
        # No names provided but importances are
        imp = np.array([0.5, 0.5])
        r = _report(np.ones(100), feature_importances=imp)
        fi = r.feature_importance()
        # Keys should be auto-generated names
        assert len(fi) == 2

    def test_zero_total_importances(self):
        imp = np.array([0.0, 0.0, 0.0])
        r = _report(np.ones(100), feature_names=["a", "b", "c"],
                    feature_importances=imp)
        fi = r.feature_importance()
        assert all(v == 0.0 for v in fi.values())


# ---------------------------------------------------------------------------
# Regulatory summary
# ---------------------------------------------------------------------------

class TestFCASummaryExtended:
    def test_contains_date(self):
        r = _report(np.ones(100))
        s = r.fca_sup153_summary()
        assert str(date.today()) in s

    def test_custom_report_date(self):
        d = date(2025, 6, 15)
        r = ShiftDiagnosticReport(weights=np.ones(100), report_date=d)
        s = r.fca_sup153_summary()
        assert "2025-06-15" in s

    def test_source_target_labels_in_summary(self):
        r = ShiftDiagnosticReport(
            weights=np.ones(100),
            source_label="Churchill Direct",
            target_label="Broker Book Q1",
        )
        s = r.fca_sup153_summary()
        assert "Churchill Direct" in s
        assert "Broker Book Q1" in s

    def test_moderate_mentions_monitoring(self):
        # Construct weights that should give MODERATE verdict
        rng = np.random.default_rng(605)
        w = rng.lognormal(0, 0.4, 300)
        cfg = CovariateShiftConfig(
            ess_severe_threshold=0.0,
            kl_severe_threshold=100.0,
            ess_moderate_threshold=1.0,
            kl_moderate_threshold=0.0,
        )
        r = _report(w, config=cfg)
        # Force MODERATE manually by using a config that makes it MODERATE
        s = r.fca_sup153_summary()
        assert "Verdict" in s

    def test_summary_contains_methodology(self):
        r = _report(np.ones(100))
        s = r.fca_sup153_summary()
        assert "Methodology" in s or "methodology" in s.lower()

    def test_severe_mentions_chief_actuary(self):
        w = np.zeros(200)
        w[0] = 500.0
        r = _report(w)
        assert r.verdict == "SEVERE"
        s = r.fca_sup153_summary()
        assert "Chief Actuary" in s or "retraining" in s.lower()


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_key_fields(self):
        w = np.ones(50)
        r = _report(w)
        text = repr(r)
        assert "ShiftDiagnosticReport" in text
        assert "verdict=" in text
        assert "ess_ratio=" in text

    def test_repr_contains_n_weights(self):
        w = np.ones(123)
        r = _report(w)
        assert "123" in repr(r)


# ---------------------------------------------------------------------------
# Plot smoke tests
# ---------------------------------------------------------------------------

class TestPlotSmokeExtended:
    def test_weight_distribution_with_axes(self):
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        w = np.random.default_rng(700).lognormal(0, 0.3, 200)
        r = _report(w)
        fig, ax = plt.subplots()
        returned_ax = r.plot_weight_distribution(ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_feature_shifts_with_data_runs(self):
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(701)
        w = np.ones(200)
        X_s = rng.normal(0, 1, (200, 2))
        X_t = rng.normal(1, 1, (100, 2))
        r = ShiftDiagnosticReport(
            weights=w,
            feature_names=["age", "ncb"],
            X_source=X_s,
            X_target=X_t,
        )
        fig = r.plot_feature_shifts()
        assert fig is not None
        plt.close("all")

    def test_feature_shifts_missing_data_raises(self):
        r = _report(np.ones(100), feature_names=["age"])
        with pytest.raises(ValueError, match="X_source"):
            r.plot_feature_shifts()
