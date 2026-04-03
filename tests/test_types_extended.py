"""Extended tests for _types module — boundary conditions and custom configs."""

import pytest
from insurance_covariate_shift._types import CovariateShiftConfig


class TestCovariateShiftConfigBoundaries:
    """Exhaustive boundary tests for the verdict logic."""

    def test_ess_just_above_severe_threshold_moderate(self):
        cfg = CovariateShiftConfig()
        # ESS just above 0.3 should not be SEVERE (assuming KL is small)
        assert cfg.verdict(ess_ratio=0.31, kl_divergence=0.05) == "MODERATE"

    def test_ess_just_above_moderate_threshold_negligible(self):
        cfg = CovariateShiftConfig()
        # ESS at exactly 0.6 -> NEGLIGIBLE (exclusive boundary)
        assert cfg.verdict(ess_ratio=0.60, kl_divergence=0.05) == "NEGLIGIBLE"

    def test_ess_just_below_moderate_threshold_moderate(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.599, kl_divergence=0.05) == "MODERATE"

    def test_kl_exactly_at_moderate_threshold_moderate(self):
        cfg = CovariateShiftConfig()
        # KL > 0.1 => MODERATE; exactly 0.1 is not > 0.1 -> depends on ESS
        # ESS = 0.9 (NEGLIGIBLE), KL = 0.1 (not > 0.1) -> NEGLIGIBLE
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.1) == "NEGLIGIBLE"

    def test_kl_just_above_moderate_threshold(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.101) == "MODERATE"

    def test_kl_exactly_at_severe_threshold_severe(self):
        cfg = CovariateShiftConfig()
        # kl_divergence >= kl_severe_threshold (0.5) -> SEVERE
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.5) == "SEVERE"

    def test_kl_just_below_severe_threshold_moderate(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.499) == "MODERATE"

    def test_both_signals_moderate_gives_moderate(self):
        cfg = CovariateShiftConfig()
        # ESS moderate AND KL moderate
        assert cfg.verdict(ess_ratio=0.5, kl_divergence=0.2) == "MODERATE"

    def test_ess_negligible_kl_moderate(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.3) == "MODERATE"

    def test_ess_one_kl_zero_negligible(self):
        # Perfect ESS, zero KL
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=1.0, kl_divergence=0.0) == "NEGLIGIBLE"

    def test_ess_zero_kl_zero_severe(self):
        # ESS=0 <= 0.3 -> SEVERE regardless of KL
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.0, kl_divergence=0.0) == "SEVERE"


class TestCustomConfig:
    def test_custom_ess_severe_boundary(self):
        cfg = CovariateShiftConfig(ess_severe_threshold=0.5)
        assert cfg.verdict(ess_ratio=0.5, kl_divergence=0.0) == "SEVERE"
        assert cfg.verdict(ess_ratio=0.51, kl_divergence=0.0) == "MODERATE"

    def test_custom_kl_thresholds(self):
        cfg = CovariateShiftConfig(kl_moderate_threshold=0.5, kl_severe_threshold=1.0)
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.4) == "NEGLIGIBLE"
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.6) == "MODERATE"
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=1.0) == "SEVERE"

    def test_very_tight_thresholds(self):
        # Tighten so almost everything is SEVERE
        cfg = CovariateShiftConfig(
            ess_severe_threshold=0.99, ess_moderate_threshold=1.0,
            kl_severe_threshold=0.001, kl_moderate_threshold=0.0005,
        )
        assert cfg.verdict(ess_ratio=0.5, kl_divergence=0.01) == "SEVERE"

    def test_dataclass_fields_accessible(self):
        cfg = CovariateShiftConfig(
            ess_severe_threshold=0.25,
            ess_moderate_threshold=0.55,
            kl_severe_threshold=0.4,
            kl_moderate_threshold=0.08,
        )
        assert cfg.ess_severe_threshold == 0.25
        assert cfg.ess_moderate_threshold == 0.55
        assert cfg.kl_severe_threshold == 0.4
        assert cfg.kl_moderate_threshold == 0.08
