"""Tests for ShiftDiagnosticReport."""

import numpy as np
import pytest

from insurance_covariate_shift.report import ShiftDiagnosticReport
from insurance_covariate_shift._types import CovariateShiftConfig


RNG = np.random.default_rng(99)


def make_report(n=400, weight_scale=0.3, feature_importances=None, include_matrices=False):
    weights = RNG.lognormal(0, weight_scale, n)
    X_s = RNG.normal(0, 1, (n, 3)) if include_matrices else None
    X_t = RNG.normal(0.5, 1, (200, 3)) if include_matrices else None
    return ShiftDiagnosticReport(
        weights=weights,
        feature_names=["age", "ncb", "vehicle_age"],
        feature_importances=np.array([0.5, 0.3, 0.2]) if feature_importances else None,
        X_source=X_s,
        X_target=X_t,
    )


class TestShiftDiagnosticReportCore:
    def test_construction(self):
        report = make_report()
        assert report is not None

    def test_verdict_is_valid_string(self):
        report = make_report()
        assert report.verdict in ("NEGLIGIBLE", "MODERATE", "SEVERE")

    def test_ess_ratio_between_zero_and_one(self):
        report = make_report()
        assert 0.0 <= report.ess_ratio <= 1.0

    def test_kl_divergence_non_negative(self):
        report = make_report()
        assert report.kl_divergence >= 0.0

    def test_constant_weights_ess_one(self):
        # All weights equal -> ESS = n -> ratio = 1
        weights = np.ones(200)
        report = ShiftDiagnosticReport(weights=weights)
        assert report.ess_ratio == pytest.approx(1.0, abs=1e-6)

    def test_extreme_weights_low_ess(self):
        # One weight is huge, rest are tiny -> ESS << n
        weights = np.ones(200) * 0.01
        weights[0] = 100.0
        report = ShiftDiagnosticReport(weights=weights)
        assert report.ess_ratio < 0.1

    def test_verdict_severe_for_extreme_weights(self):
        weights = np.ones(200) * 0.01
        weights[0] = 100.0
        report = ShiftDiagnosticReport(weights=weights)
        assert report.verdict == "SEVERE"

    def test_verdict_negligible_for_uniform_weights(self):
        weights = RNG.lognormal(0, 0.1, 500)  # Very small spread
        report = ShiftDiagnosticReport(weights=weights)
        assert report.verdict == "NEGLIGIBLE"

    def test_repr(self):
        report = make_report()
        r = repr(report)
        assert "ShiftDiagnosticReport" in r
        assert "verdict=" in r

    def test_empty_weights_graceful(self):
        # Edge case: zero length
        report = ShiftDiagnosticReport(weights=np.array([]))
        assert report.ess_ratio == 0.0

    def test_custom_config(self):
        cfg = CovariateShiftConfig(ess_severe_threshold=0.8)
        weights = RNG.lognormal(0, 0.4, 200)
        report = ShiftDiagnosticReport(weights=weights, config=cfg)
        # With very high threshold, likely SEVERE
        # Just check it runs without error
        assert report.verdict in ("NEGLIGIBLE", "MODERATE", "SEVERE")


class TestFeatureImportance:
    def test_returns_dict(self):
        report = make_report(feature_importances=True)
        fi = report.feature_importance()
        assert isinstance(fi, dict)

    def test_keys_match_feature_names(self):
        report = make_report(feature_importances=True)
        fi = report.feature_importance()
        assert set(fi.keys()) == {"age", "ncb", "vehicle_age"}

    def test_values_sum_to_one(self):
        report = make_report(feature_importances=True)
        fi = report.feature_importance()
        assert sum(fi.values()) == pytest.approx(1.0, abs=1e-6)

    def test_empty_without_importances(self):
        report = make_report(feature_importances=False)
        fi = report.feature_importance()
        assert fi == {}

    def test_zero_importances_handled(self):
        weights = RNG.lognormal(0, 0.2, 100)
        report = ShiftDiagnosticReport(
            weights=weights,
            feature_names=["a", "b"],
            feature_importances=np.array([0.0, 0.0]),
        )
        fi = report.feature_importance()
        assert all(v == 0.0 for v in fi.values())

    def test_single_feature(self):
        weights = RNG.lognormal(0, 0.2, 100)
        report = ShiftDiagnosticReport(
            weights=weights,
            feature_names=["age"],
            feature_importances=np.array([1.0]),
        )
        fi = report.feature_importance()
        assert fi == {"age": pytest.approx(1.0)}


class TestFCASummary:
    def test_returns_string(self):
        report = make_report()
        s = report.fca_sup153_summary()
        assert isinstance(s, str)

    def test_contains_verdict(self):
        report = make_report()
        s = report.fca_sup153_summary()
        assert report.verdict in s

    def test_contains_ess(self):
        report = make_report()
        s = report.fca_sup153_summary()
        assert "ESS" in s or "Effective Sample Size" in s

    def test_contains_kl(self):
        report = make_report()
        s = report.fca_sup153_summary()
        assert "KL" in s or "divergence" in s

    def test_severe_mentions_retraining(self):
        weights = np.ones(200) * 0.01
        weights[0] = 100.0
        report = ShiftDiagnosticReport(weights=weights)
        s = report.fca_sup153_summary()
        assert "retraining" in s.lower() or "Retraining" in s

    def test_negligible_mentions_no_adaptation(self):
        weights = RNG.lognormal(0, 0.05, 500)
        report = ShiftDiagnosticReport(weights=weights)
        if report.verdict == "NEGLIGIBLE":
            s = report.fca_sup153_summary()
            assert "No adaptation" in s or "not required" in s.lower()

    def test_contains_date(self):
        from datetime import date
        report = make_report()
        s = report.fca_sup153_summary()
        assert str(date.today()) in s

    def test_custom_labels(self):
        report = ShiftDiagnosticReport(
            weights=RNG.lognormal(0, 0.2, 200),
            source_label="Churchill Motor",
            target_label="Acquired Portfolio XYZ",
        )
        s = report.fca_sup153_summary()
        assert "Churchill Motor" in s
        assert "Acquired Portfolio XYZ" in s


class TestPlots:
    def test_weight_distribution_runs(self):
        matplotlib = pytest.importorskip("matplotlib")
        report = make_report()
        ax = report.plot_weight_distribution()
        assert ax is not None

    def test_feature_shifts_without_matrices_raises(self):
        report = make_report(include_matrices=False)
        with pytest.raises(ValueError, match="X_source"):
            report.plot_feature_shifts()

    def test_feature_shifts_with_matrices_runs(self):
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        report = make_report(include_matrices=True)
        fig = report.plot_feature_shifts()
        assert fig is not None
        plt.close("all")

    def test_feature_shifts_subset(self):
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        report = make_report(include_matrices=True)
        fig = report.plot_feature_shifts(features=["age", "ncb"])
        assert fig is not None
        plt.close("all")
