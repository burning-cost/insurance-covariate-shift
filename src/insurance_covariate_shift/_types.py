"""
Shared types and constants for insurance-covariate-shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "ShiftVerdict",
    "DensityRatioMethod",
    "ConformalMethod",
    "CovariateShiftConfig",
]

ShiftVerdict = Literal["NEGLIGIBLE", "MODERATE", "SEVERE"]
DensityRatioMethod = Literal["catboost", "rulsif", "kliep"]
ConformalMethod = Literal["weighted", "lrqr"]


@dataclass
class CovariateShiftConfig:
    """
    Configuration for shift detection thresholds.

    The defaults are calibrated for UK personal lines books where
    typical M&A transfers bring in a meaningfully different risk profile.

    Attributes
    ----------
    ess_severe_threshold : float
        ESS ratio below which the shift is classified SEVERE.
        Default 0.3 (30% effective sample retention).
    ess_moderate_threshold : float
        ESS ratio below which the shift is classified MODERATE.
        Default 0.6.
    kl_severe_threshold : float
        KL divergence (nats) above which shift is classified SEVERE.
        Default 0.5, roughly equivalent to two classes being well-separated.
    kl_moderate_threshold : float
        KL divergence (nats) above which shift is classified MODERATE.
        Default 0.1.
    """

    ess_severe_threshold: float = 0.3
    ess_moderate_threshold: float = 0.6
    kl_severe_threshold: float = 0.5
    kl_moderate_threshold: float = 0.1

    def verdict(self, ess_ratio: float, kl_divergence: float) -> ShiftVerdict:
        """Combine ESS and KL signals into a single verdict."""
        if ess_ratio < self.ess_severe_threshold or kl_divergence > self.kl_severe_threshold:
            return "SEVERE"
        if ess_ratio < self.ess_moderate_threshold or kl_divergence > self.kl_moderate_threshold:
            return "MODERATE"
        return "NEGLIGIBLE"
