"""
insurance-covariate-shift
=========================

Density ratio correction for insurance pricing model portability across
different book distributions.

When a pricing model trained on your direct book is deployed on an acquired
portfolio, or when a direct channel model is applied to an aggregator book,
the predictions will be biased — not because the model is wrong, but because
the distribution it was trained on differs from the one it's scoring.

This library provides:

- :class:`CovariateShiftAdaptor` — estimate p_target(x) / p_source(x) using
  CatBoost (handles postcodes and vehicle codes natively), RuLSIF, or KLIEP.
- :class:`ShiftRobustConformal` — conformal prediction intervals corrected
  for covariate shift. Coverage improves with calibration set size; see class
  docstring for implementation limitations.
- :class:`ShiftDiagnosticReport` — PS21/5 and Consumer Duty FG22/5 compatible
  documentation of the shift magnitude and drivers.

Quick start
-----------
>>> import numpy as np
>>> from insurance_covariate_shift import CovariateShiftAdaptor
>>> rng = np.random.default_rng(0)
>>> X_source = rng.normal(0, 1, (500, 3))
>>> X_target = rng.normal(0.5, 1.2, (300, 3))
>>> adaptor = CovariateShiftAdaptor(method='rulsif')
>>> adaptor.fit(X_source, X_target)
CovariateShiftAdaptor(method='rulsif')
>>> w = adaptor.importance_weights(X_source)
>>> report = adaptor.shift_diagnostic()
>>> print(report.verdict)
NEGLIGIBLE
"""

from .adaptor import CovariateShiftAdaptor
from .conformal import ShiftRobustConformal
from .density_ratio import KLIEP, RuLSIF
from .report import ShiftDiagnosticReport
from ._types import CovariateShiftConfig, ShiftVerdict

__version__ = "0.1.2"
__all__ = [
    "CovariateShiftAdaptor",
    "ShiftRobustConformal",
    "ShiftDiagnosticReport",
    "RuLSIF",
    "KLIEP",
    "CovariateShiftConfig",
    "ShiftVerdict",
]
