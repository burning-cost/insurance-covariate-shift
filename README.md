# insurance-covariate-shift

[![PyPI](https://img.shields.io/pypi/v/insurance-covariate-shift)](https://pypi.org/project/insurance-covariate-shift/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-covariate-shift)](https://pypi.org/project/insurance-covariate-shift/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()


Detect, characterise, and correct for covariate shift when deploying insurance pricing models on a different book of business.

**Blog post:** [Correcting for Covariate Shift When You Acquire an MGA Book](https://burning-cost.github.io/2026/03/13/insurance-covariate-shift/)

## What problem does this solve for a UK pricing actuary?

You trained a motor frequency model on your direct channel book. It validated well, it priced consistently, and the governance committee signed it off. Then your insurer acquired a broker portfolio. Or your business mix shifted from aggregator to price comparison. Or you're standing up a new brand targeting a different demographic.

Your model now scores a population it was never trained on. The predictions will be biased — not because the model is wrong in any fundamental sense, but because the distribution of risks has changed. Features like age, NCB, and postcode district have different distributions on the new book, and the model's fitted relationships assume the old distribution holds.

The naive fix is to retrain. But retraining takes months of data collection, validation, governance sign-off, and deployment work. In the meantime, you need to know two things:

1. **How bad is the shift, actually?** Is it large enough to matter, or are you worrying about a 2% difference in average age?
2. **Which features are driving it?** Is the new book just older, or is it a different vehicle mix, different NCD distribution, different geography?

This library answers both questions. It estimates the density ratio p_target(x) / p_source(x) — how much more (or less) likely each risk profile is in the new book compared to the training book — and uses that ratio to:

- **Classify the severity of the shift** (NEGLIGIBLE / MODERATE / SEVERE) with an Effective Sample Size ratio
- **Attribute the shift to specific features** — which covariates are the main drivers
- **Reweight evaluation metrics** so they reflect performance on the target book without needing target labels
- **Generate governance documentation** suitable for PS21/5 and Consumer Duty FG22/5

The shift diagnostic is the primary use case. It takes ~10 seconds and answers "is this book safe to deploy on?" before you have any claims data. The metric correction is a secondary tool for estimating how performance will degrade.

## Installation

```bash
uv add insurance-covariate-shift
```

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-covariate-shift/discussions). Found it useful? A star helps others find it.

## Quick start

```python
import numpy as np
from sklearn.linear_model import PoissonRegressor
from insurance_covariate_shift import CovariateShiftAdaptor

rng = np.random.default_rng(42)
# Source = direct channel, target = acquired broker book
X_source = rng.normal([35, 5, 4, 10], [8, 3, 2, 5], (2000, 4))
X_target = rng.normal([48, 7, 6, 15], [9, 3, 2, 5], (1000, 4))

adaptor = CovariateShiftAdaptor(method="catboost")
adaptor.fit(X_source, X_target, feature_names=["age", "ncb", "vehicle_age", "postcode"])

# How bad is the shift?
report = adaptor.shift_diagnostic(source_label="Direct", target_label="Acquired Broker")
print(report.verdict)          # 'NEGLIGIBLE', 'MODERATE', or 'SEVERE'
print(report.ess_ratio)        # 0.849 = no shift, 0.004 = extreme shift
print(report.kl_divergence)    # KL(target || source) in nats
print(report.feature_importance())
# {'age': 0.40, 'postcode': 0.32, 'ncb': 0.18, 'vehicle_age': 0.10}

# Regulatory text for governance committee
print(report.fca_sup153_summary())

# Importance weights for reweighting source-book metrics
weights = adaptor.importance_weights(X_source)
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

- `catboost` (default): handles high-cardinality categoricals natively. Postcode district, vehicle make-model, occupation code — CatBoost deals with these without any preprocessing. Use this for standard UK insurance tabular data. Also provides feature importance attribution.
- `rulsif`: closed-form solution, fast, no hyperparameter tuning. Use when all features are continuous (e.g. model scores, rating factors only). No feature attribution.
- `kliep`: explicitly enforces the normalisation constraint E_source[w(x)] = 1. Slower than RuLSIF but useful as a sanity check.

### ShiftDiagnosticReport

The main output of the library. This is what you take to a governance committee.

```python
report = adaptor.shift_diagnostic(source_label="Q4 2023 Direct", target_label="Broker XYZ")

print(report.verdict)          # 'NEGLIGIBLE', 'MODERATE', or 'SEVERE'
print(report.ess_ratio)        # 0 to 1; below 0.3 triggers SEVERE
print(report.kl_divergence)    # KL(target || source) in nats

# Per-feature shift attribution (CatBoost method only)
print(report.feature_importance())
# {'age': 0.40, 'postcode': 0.32, 'ncb': 0.18, 'vehicle_age': 0.10}

# Regulatory text
print(report.fca_sup153_summary())

# Plots
report.plot_weight_distribution()
report.plot_feature_shifts()
```

**Verdict thresholds:**

| Verdict | ESS ratio | KL divergence | What to do |
|---------|-----------|---------------|------------|
| NEGLIGIBLE | >= 0.60 | <= 0.10 nats | Deploy as-is |
| MODERATE | 0.30–0.60 | 0.10–0.50 nats | Apply importance weighting, monitor for 3 months |
| SEVERE | < 0.30 | > 0.50 nats | Retrain before deploying |

The SEVERE threshold (ESS < 0.3) was calibrated against actuarial practice: if less than 30% of your source sample is effectively contributing to target-distribution estimates, you are extrapolating more than interpolating.

### ShiftRobustConformal

Conformal prediction intervals that correct for distribution shift between source and target books.

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
```

**Coverage note:** Both methods provide coverage that improves with calibration set size and degrades under large shifts. Do not rely on exact finite-sample guarantees for small calibration sets (n < 2,000) or large shifts (ESS < 0.4).

## Realistic usage: weighted model evaluation

After an M&A transaction, before you retrain, you want to know how the existing model performs on the acquired book. Standard evaluation metrics computed on the source book are misleading. Importance-weighted metrics correct for the distribution shift:

```python
from sklearn.metrics import mean_absolute_error
import numpy as np
from insurance_covariate_shift import CovariateShiftAdaptor

adaptor = CovariateShiftAdaptor(method="catboost")
adaptor.fit(X_source_cal, X_target_unlabelled)
weights = adaptor.importance_weights(X_source_cal)

# Standard MAE — measures source-book performance (optimistic)
mae_source = mean_absolute_error(y_source_cal, model.predict(X_source_cal))

# Weighted MAE — estimates target-book performance without target labels
residuals = np.abs(y_source_cal - model.predict(X_source_cal))
mae_target_estimate = np.average(residuals, weights=weights)

print(f"Source MAE: {mae_source:.4f}")
print(f"Target MAE estimate: {mae_target_estimate:.4f}")
```

At n=5,000 calibration points, the importance-weighted estimate is typically 4-5x closer to the true target MAE than the unweighted estimate. At small n (< 1,000), the variance can exceed the bias correction benefit — run the diagnostic first, and if ESS < 0.3 the correction will be unreliable regardless of n.

## FCA context

Under PS21/5 (General Insurance Pricing Practices) and Consumer Duty FG22/5, insurers must demonstrate that pricing models are fair and that material changes are appropriately governed. Using a model trained on a different book distribution without adjustment is a material change that should be captured in your model change policy. The `fca_sup153_summary()` output is designed to provide the factual basis for an internal governance note.

Note: the method is named `fca_sup153_summary` for backwards API compatibility. SUP 15.3 covers material change notifications; the primary references for pricing fairness governance are PS21/5 and FG22/5.

## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library is available in [`notebooks/`](notebooks/).

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

Benchmarked on Databricks serverless, 2026-03-21. Source: direct channel (4,000 policies, age 35 mean). Three target scenarios with known ground-truth shift level.

### Part 1: Shift diagnostic accuracy

All three scenarios correctly classified. ESS drops monotonically as shift severity increases.

| Scenario | Expected | Got | ESS ratio | KL (nats) | Top feature | Result |
|----------|----------|-----|-----------|-----------|-------------|--------|
| Near-identical books | NEGLIGIBLE | NEGLIGIBLE | 0.849 | 0.09 | age (38%) | PASS |
| Broker book (age +6, urban -11pp) | MODERATE | MODERATE | 0.532 | 0.34 | age (40%) | PASS |
| Acquired MGA (age +27, NCD +4) | SEVERE | SEVERE | 0.004 | 4.55 | ncd/age | PASS |

The diagnostic takes ~1 second per scenario. The ESS spread (0.849, 0.532, 0.004) makes the severity gradient unambiguous.

Feature attribution correctly identifies the top-shifted feature in 2/3 scenarios. In the SEVERE case, age and NCD are statistically tied as main drivers (standardised shift 2.7 vs 2.7) so either top-1 answer is correct.

### Part 2: Weighted metric correction at scale (n=5,000 calibration)

With a large enough calibration set, importance-weighted MAE is substantially closer to the oracle (true target MAE) than the unweighted estimate.

| Metric | Value | Notes |
|--------|-------|-------|
| Source calibration MAE | 0.0636 | What the model reports on its own data |
| Importance-weighted MAE | 0.0552 | Our estimate of target performance |
| Oracle target MAE | 0.0528 | True answer, normally unknown |

IW estimation error vs oracle: 0.0024 (4.5x better than unweighted error of 0.0107).

**When IW correction helps:** moderate shift (ESS 0.3–0.6) with n_calibration >= 2,000. At small n or severe shift (ESS < 0.3), the variance in the weighted estimator dominates and the correction may not improve accuracy.

**When to use this library:**
- Run the diagnostic immediately when standing up on any new book of business
- Use the verdict and feature attribution for governance documentation
- Use importance-weighted metrics to estimate performance degradation before claims emerge
- Set a monitoring flag if verdict is SEVERE — schedule a retraining sprint
