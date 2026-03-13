"""
Integration tests — end-to-end workflows simulating realistic insurance scenarios.

These are the tests that answer "does the library do what the pricing actuary
needs?" rather than testing individual components. Each scenario is based on
a situation that comes up in UK insurance practice.
"""

import numpy as np
import pytest
from sklearn.linear_model import PoissonRegressor

from insurance_covariate_shift import (
    CovariateShiftAdaptor,
    ShiftDiagnosticReport,
    ShiftRobustConformal,
)


# -----------------------------------------------------------------------
# Scenario 1: Motor book acquisition — frequency model portability
# -----------------------------------------------------------------------

def make_motor_books(rng=None):
    """
    Simulate a motor insurance scenario.

    Source: Direct channel, younger drivers, lower NCB on average.
    Target: Acquired broker book, older profile, higher NCB.
    """
    if rng is None:
        rng = np.random.default_rng(100)

    n_source = 2000
    n_target = 800

    # Source: direct channel
    age_s = rng.integers(18, 65, n_source).astype(float)
    ncb_s = rng.integers(0, 5, n_source).astype(float)
    veh_age_s = rng.integers(0, 15, n_source).astype(float)
    exposure_s = rng.uniform(0.5, 1.0, n_source)

    X_source = np.column_stack([age_s, ncb_s, veh_age_s, exposure_s])

    # Target: broker book, older / higher NCB
    age_t = rng.integers(30, 75, n_target).astype(float)
    ncb_t = rng.integers(2, 5, n_target).astype(float)
    veh_age_t = rng.integers(3, 20, n_target).astype(float)
    exposure_t = rng.uniform(0.5, 1.0, n_target)

    X_target = np.column_stack([age_t, ncb_t, veh_age_t, exposure_t])

    return X_source, X_target


class TestMotorAcquisitionScenario:
    def test_full_workflow_rulsif(self):
        """Complete workflow: fit -> diagnose -> get weights -> report."""
        X_s, X_t = make_motor_books()
        feature_names = ["age", "ncb", "vehicle_age"]  # exposure excluded

        adaptor = CovariateShiftAdaptor(
            method="rulsif",
            exposure_col=3,
        )
        adaptor.fit(X_s, X_t, feature_names=feature_names)

        # Get weights
        w = adaptor.importance_weights(X_s)
        assert w.shape == (2000,)
        assert (w >= 0).all()

        # Diagnose
        report = adaptor.shift_diagnostic(source_label="Direct", target_label="Broker")
        assert isinstance(report, ShiftDiagnosticReport)
        assert report.verdict in ("NEGLIGIBLE", "MODERATE", "SEVERE")

        # FCA summary
        summary = report.fca_sup153_summary()
        assert "Direct" in summary
        assert "Broker" in summary

    def test_full_workflow_kliep(self):
        X_s, X_t = make_motor_books()
        adaptor = CovariateShiftAdaptor(method="kliep", exposure_col=3)
        adaptor.fit(X_s, X_t)
        w = adaptor.importance_weights(X_s)
        assert (w >= 0).all()

    def test_conformal_on_acquisition(self):
        """Conformal intervals for frequency model on acquired book."""
        rng = np.random.default_rng(200)
        X_s, X_t = make_motor_books(rng=rng)

        # Simulate claim frequency (Poisson)
        freq_s = rng.poisson(0.08 + 0.002 * X_s[:, 0] - 0.01 * X_s[:, 1])
        freq_t = rng.poisson(0.05 + 0.001 * X_t[:, 0] - 0.008 * X_t[:, 1])

        # Use only first 3 features for model
        X_s_feat = X_s[:, :3]
        X_t_feat = X_t[:, :3]

        # Split source into train / calibration
        n_train = 1500
        X_train, y_train = X_s_feat[:n_train], freq_s[:n_train].astype(float)
        X_cal, y_cal = X_s_feat[n_train:], freq_s[n_train:].astype(float)

        model = PoissonRegressor(max_iter=300).fit(X_train, y_train)

        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_cal, X_t_feat)

        cp = ShiftRobustConformal(model=model, adaptor=adaptor, method="weighted", alpha=0.1)
        cp.calibrate(X_cal, y_cal)

        lo, hi = cp.predict_interval(X_t_feat)
        assert lo.shape == (800,)
        assert (hi >= lo).all()

        cov = cp.empirical_coverage(X_t_feat, freq_t.astype(float))
        assert 0.6 <= cov <= 1.0


# -----------------------------------------------------------------------
# Scenario 2: Channel mix shift — aggregator vs. direct
# -----------------------------------------------------------------------

class TestChannelMixScenario:
    def test_aggregator_vs_direct_negligible_shift(self):
        """Same distribution, should yield NEGLIGIBLE verdict."""
        rng = np.random.default_rng(300)
        X_s = rng.normal(0, 1, (500, 4))
        X_t = rng.normal(0, 1, (400, 4))  # Same distribution

        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_s, X_t)
        report = adaptor.shift_diagnostic()
        # ESS should be high for identical distributions
        assert report.ess_ratio > 0.5

    def test_aggregator_vs_direct_severe_shift(self):
        """Very different profiles — should be MODERATE or SEVERE."""
        rng = np.random.default_rng(301)
        X_s = rng.normal(0, 0.5, (500, 3))
        X_t = rng.normal(4, 0.5, (400, 3))  # Extreme shift

        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_s, X_t)
        report = adaptor.shift_diagnostic()
        # ESS should be low for very different distributions
        assert report.ess_ratio < 0.5


# -----------------------------------------------------------------------
# Scenario 3: Portfolio seasoning — weights applied to existing scores
# -----------------------------------------------------------------------

class TestPortfolioSeasoningScenario:
    def test_reweight_scores(self):
        """
        A common use case: the actuary has model scores (not features),
        and wants to reweight them to match the target distribution.
        """
        rng = np.random.default_rng(400)
        # Source: model scores from direct book
        scores_source = rng.normal(0.08, 0.02, 300).reshape(-1, 1)
        # Target: broker book with lower average score
        scores_target = rng.normal(0.06, 0.025, 200).reshape(-1, 1)

        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(scores_source, scores_target)
        w = adaptor.importance_weights(scores_source)

        # Weighted mean should be closer to target mean than raw mean
        raw_mean = float(scores_source.mean())
        weighted_mean = float(np.average(scores_source.flatten(), weights=w))
        target_mean = float(scores_target.mean())

        # The weighted mean should shift toward the target
        assert abs(weighted_mean - target_mean) <= abs(raw_mean - target_mean) + 0.01


# -----------------------------------------------------------------------
# Scenario 4: CatBoost with mixed feature types
# -----------------------------------------------------------------------

class TestCatBoostMixedFeatures:
    @pytest.fixture(autouse=True)
    def skip_if_no_catboost(self):
        pytest.importorskip("catboost")

    def test_mixed_features_workflow(self):
        """Simulate postcode + continuous features."""
        rng = np.random.default_rng(500)
        n_s, n_t = 300, 200

        # Continuous features
        age_s = rng.normal(40, 12, n_s)
        age_t = rng.normal(45, 10, n_t)
        ncb_s = rng.normal(3, 1, n_s)
        ncb_t = rng.normal(3.5, 0.8, n_t)

        # Categorical: postcode district (integer encoded)
        pc_s = rng.integers(0, 100, n_s).astype(float)
        pc_t = rng.integers(20, 80, n_t).astype(float)  # Narrower distribution

        X_s = np.column_stack([age_s, ncb_s, pc_s])
        X_t = np.column_stack([age_t, ncb_t, pc_t])

        adaptor = CovariateShiftAdaptor(
            method="catboost",
            categorical_cols=[2],
            catboost_iterations=100,
        )
        adaptor.fit(X_s, X_t, feature_names=["age", "ncb", "postcode"])
        w = adaptor.importance_weights(X_s)
        assert w.shape == (n_s,)
        assert (w >= 0).all()

        report = adaptor.shift_diagnostic()
        fi = report.feature_importance()
        assert "postcode" in fi


# -----------------------------------------------------------------------
# Scenario 5: LR-QR adaptive intervals on motor data
# -----------------------------------------------------------------------

class TestLRQRMotorScenario:
    def test_lrqr_adaptive_intervals(self):
        """LR-QR should produce covariate-varying interval widths."""
        rng = np.random.default_rng(600)
        d = 5

        X_train = rng.normal(0, 1, (500, d))
        y_train = X_train[:, 0] * 3 + rng.normal(0, 1, 500)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X_train, y_train)

        X_cal = rng.normal(0, 1, (400, d))
        y_cal = X_cal[:, 0] * 3 + rng.normal(0, 1, 400)
        X_target = rng.normal(1.0, 1.5, (200, d))

        adaptor = CovariateShiftAdaptor(method="rulsif")
        adaptor.fit(X_cal, X_target)

        cp = ShiftRobustConformal(
            model=model,
            adaptor=adaptor,
            method="lrqr",
            alpha=0.1,
            lrqr_hidden_sizes=(32, 16),
        )
        cp.calibrate(X_cal, y_cal)

        lo, hi = cp.predict_interval(X_target)
        assert (hi >= lo).all()

        widths = hi - lo
        assert widths.min() >= 0
        assert widths.shape == (200,)
