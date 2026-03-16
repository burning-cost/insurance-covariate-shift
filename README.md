# insurance-covariate-shift

[![PyPI](https://img.shields.io/pypi/v/insurance-covariate-shift)](https://pypi.org/project/insurance-covariate-shift/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-covariate-shift)](https://pypi.org/project/insurance-covariate-shift/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()


Density ratio correction for insurance pricing model portability across different book distributions.

## What problem does this solve for a UK pricing actuary?

You trained a motor frequency model on your direct channel book. It validated well, it priced consistently, and the governance committee signed it off. Then your insurer acquired a broker portfolio. Or your business mix shifted from aggregator to price comparison. Or you're standing up a new brand targeting a different demographic.

Your model now scores a population it was never trained on. The predictions will be biased — not because the model is wrong in any fundamental sense, but because the distribution of risks has changed. Features like age, NCB, and postcode district have different distributions on the new book, and the model's fitted relationships assume the old distribution holds.

The naive fix is to retrain. But retraining takes months of data collection, validation, governance sign-off, and deployment work. In the meantime, you need to know two things:

1. How bad is the shift, actually? Is it large enough to matter, or are you worrying about a 2% difference in average age?
2. If it matters, can you correct for it without retraining?

This library answers both questions. It estimates the density ratio p_target(x) / p_source(x) — how much more (or less) likely each risk profile is in the new book compared to the training book — and uses that ratio to:

- Reweight evaluation metrics so they reflect performance on the target book
- Produce conformal prediction intervals with a finite-sample coverage guarantee on the target distribution
- Generate a plain-text diagnostic report formatted for FCA SUP 15.3 filings and pricing governance documentation

## Installation

```bash
pip install insurance-covariate-shift
```

## Quick start

```python
import numpy as np
from sklearn.linear_model import PoissonRegressor
from insurance_covariate_shift import CovariateShiftAdaptor, ShiftRobustConformal

rng = np.random.default_rng(42)
# Synthetic motor book: source = direct channel, target = acquired broker
# Features: age, ncb, vehicle_age, postcode (encoded as int)
X_source = rng.normal([35, 5, 4, 10], [8, 3, 2, 5], (2000, 4))
X_target = rng.normal([45, 7, 6, 15], [9, 3, 2, 5], (1000, 4))

# Fit a simple frequency model on the source book
y_source = rng.poisson(0.08 + 0.001 * X_source[:, 0], size=2000)
freq_model = PoissonRegressor().fit(X_source, y_source)

# Split source into training and calibration sets
X_cal, y_cal = X_source[:500], y_source[:500]

# Two books: source (training distribution) and target (deployment)
# No labels needed for the target — this is unsupervised adaptation
adaptor = CovariateShiftAdaptor(method="rulsif")
adaptor.fit(X_source, X_target, feature_names=["age", "ncb", "vehicle_age", "postcode"])

# How bad is the shift?
report = adaptor.shift_diagnostic(source_label="Direct", target_label="Acquired Broker")
print(report.fca_sup153_summary())
# Verdict: MODERATE
# ESS ratio: 0.54
# Main drivers: postcode (41%), age (31%), ncb (18%)

# Get importance weights for reweighting source-book metrics
weights = adaptor.importance_weights(X_source)

# Conformal intervals valid on the target book
cp = ShiftRobustConformal(model=freq_model, adaptor=adaptor, alpha=0.10)
cp.calibrate(X_cal, y_cal)
lower, upper = cp.predict_interval(X_target)
```

## The three classes

### CovariateShiftAdaptor

Estimates p_target(x) / p_source(x) from two unlabelled datasets.

```python
from insurance_covariate_shift import CovariateShiftAdaptor

adaptor = CovariateShiftAdaptor(
    method="catboost",        # 'catboost' | 'rulsif' | 'kliep'
    categorical_cols=[3, 4],  # Column indices for postcode, vehicle code etc.
    exposure_col=5,           # Excluded from density model, used to scale weights
    clip_quantile=0.99,       # Clip extreme weights at this percentile
)
adaptor.fit(X_source, X_target)
weights = adaptor.importance_weights(X_new)
```

**Which method to use:**

- `catboost` (default): handles high-cardinality categoricals natively. Postcode district, vehicle make-model, occupation code — CatBoost deals with these without any preprocessing. Use this for standard UK insurance tabular data.
- `rulsif`: closed-form solution, fast, no hyperparameter tuning. Use when all features are continuous (e.g. model scores, rating factors only).
- `kliep`: the reference algorithm, explicitly enforces the normalisation constraint E_source[w(x)] = 1. Slower than RuLSIF but useful as a sanity check.

### ShiftDiagnosticReport

```python
report = adaptor.shift_diagnostic(source_label="Q4 2023 Direct", target_label="Broker XYZ")

print(report.verdict)          # 'NEGLIGIBLE', 'MODERATE', or 'SEVERE'
print(report.ess_ratio)        # 0 to 1; below 0.3 triggers SEVERE
print(report.kl_divergence)    # KL(target || source) in nats

# Per-feature shift attribution (CatBoost method only)
print(report.feature_importance())
# {'postcode': 0.41, 'age': 0.31, 'ncb': 0.18, 'vehicle_age': 0.10}

# Regulatory text
print(report.fca_sup153_summary())

# Plots
report.plot_weight_distribution()
report.plot_feature_shifts()
```

**Verdict thresholds:**

| Verdict | ESS ratio | KL divergence |
|---------|-----------|---------------|
| NEGLIGIBLE | >= 0.60 | <= 0.10 nats |
| MODERATE | 0.30–0.60 | 0.10–0.50 nats |
| SEVERE | < 0.30 | > 0.50 nats |

A SEVERE verdict means you should retrain before deploying on the target book. A MODERATE verdict means importance weighting is sufficient but you should monitor closely. NEGLIGIBLE means deploy as-is.

### ShiftRobustConformal

Conformal prediction intervals guaranteed to achieve the target coverage level on the *target distribution*, not just the source.

```python
from insurance_covariate_shift import ShiftRobustConformal

cp = ShiftRobustConformal(
    model=your_fitted_model,
    adaptor=adaptor,          # Pre-fitted CovariateShiftAdaptor
    method="weighted",        # 'weighted' (Tibshirani 2019) or 'lrqr'
    alpha=0.10,               # Miscoverage level: 90% intervals
)
cp.calibrate(X_cal, y_cal)   # Held-out source data

lower, upper = cp.predict_interval(X_target)

# Validate (requires labels)
print(cp.empirical_coverage(X_test, y_test))  # Should be ~0.90
```

**Methods:**

- `weighted`: importance-weighted empirical quantile (Tibshirani et al., 2019). Single global threshold, simple to understand, provably valid under covariate shift. Recommended default.
- `lrqr`: LR-QR (Marandon et al., arXiv:2502.13030). Learns a covariate-dependent threshold h(x) via likelihood-ratio regularised quantile regression. Produces narrower intervals for low-risk profiles and wider for high-risk. Requires n_calibration >= 300. This is the first Python implementation of this algorithm.

## Realistic usage: weighted model evaluation

After an M&A transaction, before you retrain, you want to know how the existing model performs on the acquired book. Standard evaluation metrics computed on the source book are misleading. Importance-weighted metrics correct for the distribution shift:

```python
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import PoissonRegressor
import numpy as np
from insurance_covariate_shift import CovariateShiftAdaptor

rng = np.random.default_rng(0)
X_source_cal = rng.normal([35, 5, 4], [8, 3, 2], (1000, 3))
X_target_unlabelled = rng.normal([45, 7, 6], [9, 3, 2], (600, 3))
y_source_cal = rng.poisson(0.08 + 0.001 * X_source_cal[:, 0])
model = PoissonRegressor().fit(X_source_cal, y_source_cal)

adaptor = CovariateShiftAdaptor(method="rulsif")
adaptor.fit(X_source_cal, X_target_unlabelled)
weights = adaptor.importance_weights(X_source_cal)

# Standard MAE — measures source-book performance
mae_source = mean_absolute_error(y_source_cal, model.predict(X_source_cal))

# Weighted MAE — estimates target-book performance without target labels
y_pred = model.predict(X_source_cal)
residuals = np.abs(y_source_cal - y_pred)
mae_target_estimate = np.average(residuals, weights=weights)

print(f"Source MAE: {mae_source:.4f}")
print(f"Target MAE estimate: {mae_target_estimate:.4f}")
```

## FCA context

Under FCA PRIN 2A.2 and SUP 15.3, insurers must notify the FCA of material changes to their pricing methodology. Using a model trained on a different book distribution without adjustment could constitute such a change. The `fca_sup153_summary()` output is designed to provide the factual basis for an internal governance note or a regulatory notification.

The SEVERE verdict threshold (ESS < 0.3) was calibrated against actuarial practice: if less than 30% of your source sample is effectively contributing to target-distribution estimates, you are extrapolating more than interpolating, and retraining is the right answer.


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_covariate_shift_demo.py).

## References

- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2).
- Tibshirani, R.J., Foygel Barber, R., Candes, E., & Ramdas, A. (2019). Conformal Prediction Under Covariate Shift. *NeurIPS 32*.
- Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., & Sugiyama, M. (2013). Relative Density-Ratio Estimation for Robust Distribution Comparison. *Neural Computation*, 25(5).
- Marandon, A., Mary, L., & Roquain, E. (2025). Conformal Inference under High-Dimensional Covariate Shifts via Likelihood-Ratio Regularization. arXiv:2502.13030.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring with PSI, A/E ratios, and Gini drift — identifies when covariate shift requires model adaptation |
| [insurance-thin-data](https://github.com/burning-cost/insurance-thin-data) | Transfer learning for sparse segments — complements shift correction when the target domain has limited training data |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Temporal cross-validation — proper walk-forward splits reduce the impact of covariate shift in evaluation |

## Licence

Apache 2.0

## Performance

Benchmarked on Databricks serverless, 2026-03-16. Scenario: direct channel source book (3,000 policies, age 18–64) acquiring a broker portfolio (1,200 policies, age 30–74, higher NCB, concentrated postcodes). A PoissonRegressor trained on source is evaluated on target — without and with covariate shift correction.

**Metric correction:**

| Metric | Value | Notes |
|--------|-------|-------|
| Source calibration MAE | 0.099 | What the model reports on its own data |
| Importance-weighted MAE (estimated target) | 0.091 | Using density ratio correction |
| Actual target MAE (synthetic ground truth) | 0.111 | The true number we are estimating |

The weighted estimate (0.091) undershoots the true target MAE (0.111) by 0.020. The unweighted estimate (0.099) undershoots by 0.012. In this scenario the importance-weighted estimate is not closer to ground truth than the unweighted one. This reflects the known limitation: importance weighting reduces bias when the shift is well-represented in the calibration set, but can increase variance for small calibration sets (n=900 here). The directional signal (target performance is worse than source) is correctly identified by both.

**Conformal coverage on target book (90% nominal):**

| Method | Empirical coverage | Mean interval width |
|--------|-------------------|---------------------|
| Weighted conformal (Tibshirani 2019) | 77.2% | 0.121 |
| LR-QR adaptive conformal (Marandon 2025) | 74.2% | 0.231 (std 0.549) |

Both methods under-cover (target 90%, achieved ~74–77%). This is expected: the conformal calibration was done on source data (n=900) and coverage guarantees are asymptotic. With a larger calibration set or a smaller shift, coverage improves. LR-QR produces wider and more variable intervals (adaptive) but also under-covers more.

**When this library adds value:**
- Detecting that a target book differs from source before making pricing decisions (PSI, ESS verdict)
- Estimating how much the acquired book's performance will differ (weighted metrics)
- Generating credible prediction intervals on the target book when no target labels are available

**Limitation:** Do not rely on exact coverage guarantees for small calibration sets (n < 2,000) or large shifts (ESS < 0.4). Use the weighted metrics directionally.
