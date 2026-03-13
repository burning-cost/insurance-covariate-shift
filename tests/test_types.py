"""Tests for _types module."""

import pytest
from insurance_covariate_shift._types import CovariateShiftConfig


class TestCovariateShiftConfig:
    def test_defaults(self):
        cfg = CovariateShiftConfig()
        assert cfg.ess_severe_threshold == 0.3
        assert cfg.ess_moderate_threshold == 0.6
        assert cfg.kl_severe_threshold == 0.5
        assert cfg.kl_moderate_threshold == 0.1

    def test_verdict_negligible(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.05) == "NEGLIGIBLE"

    def test_verdict_moderate_by_ess(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.45, kl_divergence=0.05) == "MODERATE"

    def test_verdict_moderate_by_kl(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.2) == "MODERATE"

    def test_verdict_severe_by_ess(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.1, kl_divergence=0.05) == "SEVERE"

    def test_verdict_severe_by_kl(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.9, kl_divergence=0.8) == "SEVERE"

    def test_verdict_boundary_ess_severe(self):
        cfg = CovariateShiftConfig()
        # Exactly at boundary — boundary belongs to SEVERE
        assert cfg.verdict(ess_ratio=0.3, kl_divergence=0.05) == "SEVERE"

    def test_verdict_boundary_ess_moderate(self):
        cfg = CovariateShiftConfig()
        assert cfg.verdict(ess_ratio=0.6, kl_divergence=0.05) == "NEGLIGIBLE"

    def test_custom_thresholds(self):
        cfg = CovariateShiftConfig(ess_severe_threshold=0.5, ess_moderate_threshold=0.8)
        assert cfg.verdict(ess_ratio=0.6, kl_divergence=0.05) == "MODERATE"

    def test_severe_beats_all(self):
        cfg = CovariateShiftConfig()
        # Low ESS AND high KL — still SEVERE
        assert cfg.verdict(ess_ratio=0.05, kl_divergence=1.0) == "SEVERE"
