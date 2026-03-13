# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-covariate-shift: Full Workflow Demo
# MAGIC
# MAGIC **Scenario**: A UK motor insurer has trained a claims frequency model on
# MAGIC their direct channel book. They have just acquired a broker portfolio with
# MAGIC a meaningfully different risk profile — older drivers, higher average NCB,
# MAGIC different postcode spread.
# MAGIC
# MAGIC This notebook walks through:
# MAGIC 1. Generating realistic synthetic motor insurance data
# MAGIC 2. Detecting and quantifying the distribution shift
# MAGIC 3. Producing a FCA SUP 15.3 diagnostic report
# MAGIC 4. Computing conformal prediction intervals valid on the target book
# MAGIC 5. Comparing weighted vs. unweighted interval coverage

# COMMAND ----------

# MAGIC %pip install insurance-covariate-shift catboost --quiet

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.model_selection import train_test_split

from insurance_covariate_shift import (
    CovariateShiftAdaptor,
    ShiftRobustConformal,
    ShiftDiagnosticReport,
)

print("insurance-covariate-shift loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK Motor Insurance Data
# MAGIC
# MAGIC We generate two books with realistic feature distributions:
# MAGIC - **Source (Direct)**: 18-65 age range, 0-5 NCB, broad postcode spread
# MAGIC - **Target (Broker acquisition)**: 30-75 age range, higher NCB, concentrated postcodes
# MAGIC
# MAGIC The target book is meaningfully different but not absurdly so — this is a
# MAGIC realistic M&A scenario where the acquired book looks like a more experienced
# MAGIC and urban clientele.

# COMMAND ----------

rng = np.random.default_rng(42)

# ---- Source book (direct channel) ------------------------------------
n_source = 3000

age_s = rng.integers(18, 65, n_source).astype(float)
ncb_s = rng.integers(0, 5, n_source).astype(float)
vehicle_age_s = rng.integers(0, 15, n_source).astype(float)
postcode_region_s = rng.integers(0, 50, n_source).astype(float)  # 50 regions
exposure_s = rng.uniform(0.3, 1.0, n_source)

# True log-frequency: age peaks mid-30s, NCB reduces risk, urban (low postcode) higher
log_lambda_s = (
    -2.5
    + 0.005 * (age_s - 18)
    - 0.002 * (age_s - 40) ** 2 / 100
    - 0.08 * ncb_s
    + 0.02 * (vehicle_age_s > 8).astype(float)
    + 0.003 * (postcode_region_s < 10).astype(float)
)
lambda_s = np.exp(log_lambda_s) * exposure_s
claims_s = rng.poisson(lambda_s)

X_source = np.column_stack([age_s, ncb_s, vehicle_age_s, postcode_region_s])
y_source = claims_s.astype(float)

print(f"Source book: {n_source} risks")
print(f"  Age:             mean={age_s.mean():.1f}, std={age_s.std():.1f}")
print(f"  NCB:             mean={ncb_s.mean():.2f}")
print(f"  Claim frequency: {(y_source > 0).mean():.3f}")

# COMMAND ----------

# ---- Target book (acquired broker portfolio) -------------------------
n_target = 1200

age_t = rng.integers(30, 75, n_target).astype(float)   # Older
ncb_t = rng.integers(2, 5, n_target).astype(float)     # Higher NCB
vehicle_age_t = rng.integers(2, 18, n_target).astype(float)
postcode_region_t = rng.integers(5, 25, n_target).astype(float)  # Narrower spread, urban
exposure_t = rng.uniform(0.3, 1.0, n_target)

log_lambda_t = (
    -2.5
    + 0.005 * (age_t - 18)
    - 0.002 * (age_t - 40) ** 2 / 100
    - 0.08 * ncb_t
    + 0.02 * (vehicle_age_t > 8).astype(float)
    + 0.003 * (postcode_region_t < 10).astype(float)
)
lambda_t = np.exp(log_lambda_t) * exposure_t
claims_t = rng.poisson(lambda_t)

X_target = np.column_stack([age_t, ncb_t, vehicle_age_t, postcode_region_t])
y_target = claims_t.astype(float)

print(f"\nTarget book (acquired): {n_target} risks")
print(f"  Age:             mean={age_t.mean():.1f}, std={age_t.std():.1f}")
print(f"  NCB:             mean={ncb_t.mean():.2f}")
print(f"  Claim frequency: {(y_target > 0).mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Detect and Quantify the Shift
# MAGIC
# MAGIC We fit a CatBoost classifier to distinguish source from target observations.
# MAGIC The density ratio weights tell us how much to upweight each source observation
# MAGIC to make it "look like" the target book.

# COMMAND ----------

adaptor = CovariateShiftAdaptor(
    method="catboost",
    categorical_cols=[1, 3],   # NCB (ordinal) and postcode region
    exposure_col=None,          # Not passing exposure here
    clip_quantile=0.99,
    catboost_iterations=300,
)

adaptor.fit(
    X_source,
    X_target,
    feature_names=["age", "ncb", "vehicle_age", "postcode_region"],
)

weights = adaptor.importance_weights(X_source)
print(f"Importance weights: min={weights.min():.3f}, mean={weights.mean():.3f}, max={weights.max():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diagnostic Report

# COMMAND ----------

report = adaptor.shift_diagnostic(
    source_label="Direct Channel 2023",
    target_label="Acquired Broker Book",
)

print(report.fca_sup153_summary())

# COMMAND ----------

# Visualise weight distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

report.plot_weight_distribution(ax=axes[0])

# Feature importance bar chart
fi = report.feature_importance()
features = list(fi.keys())
scores = list(fi.values())
axes[1].bar(features, scores, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], alpha=0.8)
axes[1].set_title("Feature Contribution to Shift")
axes[1].set_ylabel("Normalised importance")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Feature marginal distributions
fig = report.plot_feature_shifts(figsize_per_panel=(4.5, 3.5))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train a Frequency Model on Source Book
# MAGIC
# MAGIC We split the source book into training and calibration sets. The calibration
# MAGIC set is withheld from model fitting — it is used only for conformal calibration.

# COMMAND ----------

X_train, X_cal, y_train, y_cal = train_test_split(
    X_source, y_source, test_size=0.3, random_state=42
)

# Poisson GLM for frequency
freq_model = PoissonRegressor(alpha=0.01, max_iter=500)
freq_model.fit(X_train, y_train)

# Source-book performance
y_pred_source = freq_model.predict(X_cal)
mae_source = np.mean(np.abs(y_cal - y_pred_source))
print(f"Source calibration MAE: {mae_source:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Weighted vs. Unweighted Metric Comparison
# MAGIC
# MAGIC The key question before the acquisition closes: how does my model
# MAGIC actually perform on the target book? We estimate this using importance
# MAGIC weighting — no target labels needed.

# COMMAND ----------

# Fit adaptor on calibration set vs target
adaptor_cal = CovariateShiftAdaptor(method="catboost", catboost_iterations=200)
adaptor_cal.fit(X_cal, X_target, feature_names=["age", "ncb", "vehicle_age", "postcode_region"])
cal_weights = adaptor_cal.importance_weights(X_cal)

# Standard MAE
residuals = np.abs(y_cal - freq_model.predict(X_cal))
mae_unweighted = float(np.mean(residuals))

# Weighted MAE — estimates target-book performance
mae_weighted = float(np.average(residuals, weights=cal_weights))

# Actual target MAE (available here because it's synthetic)
y_pred_target = freq_model.predict(X_target)
mae_actual_target = float(np.mean(np.abs(y_target - y_pred_target)))

print(f"Unweighted source MAE:         {mae_unweighted:.4f}  (optimistic)")
print(f"Importance-weighted MAE:       {mae_weighted:.4f}  (estimated target)")
print(f"Actual target MAE (synthetic): {mae_actual_target:.4f}  (ground truth)")
print()
print("The weighted estimate should be closer to the actual target MAE.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Conformal Prediction Intervals on the Target Book
# MAGIC
# MAGIC We construct prediction intervals using two methods:
# MAGIC - **Weighted** (Tibshirani 2019): fast, single global threshold
# MAGIC - **LR-QR** (Marandon 2025): adaptive per-observation threshold

# COMMAND ----------

# Method 1: Weighted conformal
cp_weighted = ShiftRobustConformal(
    model=freq_model,
    adaptor=adaptor_cal,
    method="weighted",
    alpha=0.10,
)
cp_weighted.calibrate(X_cal, y_cal)

lo_w, hi_w = cp_weighted.predict_interval(X_target)
cov_weighted = cp_weighted.empirical_coverage(X_target, y_target)
width_weighted = float((hi_w - lo_w).mean())

print("=== Weighted conformal (Tibshirani 2019) ===")
print(f"  Empirical coverage: {cov_weighted:.3f}  (target: 0.90)")
print(f"  Mean interval width: {width_weighted:.4f}")

# COMMAND ----------

# Method 2: LR-QR adaptive conformal
# Requires a reasonably large calibration set
adaptor_lrqr = CovariateShiftAdaptor(method="catboost", catboost_iterations=200)
adaptor_lrqr.fit(X_cal, X_target)

cp_lrqr = ShiftRobustConformal(
    model=freq_model,
    adaptor=adaptor_lrqr,
    method="lrqr",
    alpha=0.10,
    lrqr_lambda=1.0,
    lrqr_hidden_sizes=(64, 32),
)
cp_lrqr.calibrate(X_cal, y_cal)

lo_l, hi_l = cp_lrqr.predict_interval(X_target)
cov_lrqr = cp_lrqr.empirical_coverage(X_target, y_target)
width_lrqr_mean = float((hi_l - lo_l).mean())
width_lrqr_std = float((hi_l - lo_l).std())

print("=== LR-QR adaptive conformal (Marandon 2025) ===")
print(f"  Empirical coverage: {cov_lrqr:.3f}  (target: 0.90)")
print(f"  Mean interval width: {width_lrqr_mean:.4f}")
print(f"  Std of interval widths: {width_lrqr_std:.4f}  (> 0 means adaptive)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coverage comparison plot

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Weighted: intervals for first 100 target risks
idx = np.argsort(X_target[:, 0])[:100]  # Sort by age for visual clarity
ax = axes[0]
ax.errorbar(
    np.arange(100),
    (lo_w[idx] + hi_w[idx]) / 2,
    yerr=(hi_w[idx] - lo_w[idx]) / 2,
    fmt="none",
    alpha=0.3,
    color="#1f77b4",
    label="Interval",
)
ax.scatter(np.arange(100), y_target[idx], s=8, color="crimson", zorder=5, label="Actual")
ax.set_title(f"Weighted Conformal (cov={cov_weighted:.2f})")
ax.set_xlabel("Risk (sorted by age)")
ax.set_ylabel("Claims")
ax.legend()

# LR-QR: show interval widths vary
ax = axes[1]
widths_lrqr = hi_l - lo_l
ax.scatter(X_target[:, 0], widths_lrqr, s=5, alpha=0.4, color="#2ca02c")
ax.set_xlabel("Age")
ax.set_ylabel("Interval width")
ax.set_title(f"LR-QR Adaptive Widths (cov={cov_lrqr:.2f})")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Shift verdict | MODERATE |
# MAGIC | ESS ratio | ~0.5 |
# MAGIC | Main driver | NCB, postcode region |
# MAGIC | Weighted coverage (target) | ~0.90 |
# MAGIC | LR-QR coverage (target) | ~0.90 |
# MAGIC
# MAGIC **Recommendation**: Apply importance weighting when evaluating model
# MAGIC performance on the acquired book. Deploy with conformal intervals using
# MAGIC the weighted method. Retrain once 12 months of target-book claims data
# MAGIC has accumulated.
# MAGIC
# MAGIC ---
# MAGIC *Generated by insurance-covariate-shift v0.1.0 — https://github.com/burning-cost/insurance-covariate-shift*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: RuLSIF and KLIEP methods
# MAGIC
# MAGIC For situations where all features are continuous (model scores, rating factors),
# MAGIC RuLSIF and KLIEP provide closed-form alternatives.

# COMMAND ----------

# RuLSIF on continuous features only (drop categorical postcode)
from insurance_covariate_shift import CovariateShiftAdaptor

X_s_cont = X_source[:, :3]  # age, ncb (treated as continuous), vehicle_age
X_t_cont = X_target[:, :3]

adaptor_rulsif = CovariateShiftAdaptor(method="rulsif")
adaptor_rulsif.fit(X_s_cont, X_t_cont)
w_rulsif = adaptor_rulsif.importance_weights(X_s_cont)

adaptor_kliep = CovariateShiftAdaptor(method="kliep")
adaptor_kliep.fit(X_s_cont, X_t_cont)
w_kliep = adaptor_kliep.importance_weights(X_s_cont)

print("RuLSIF weights:  mean={:.3f}, std={:.3f}".format(w_rulsif.mean(), w_rulsif.std()))
print("KLIEP weights:   mean={:.3f}, std={:.3f}".format(w_kliep.mean(), w_kliep.std()))

report_rulsif = adaptor_rulsif.shift_diagnostic()
report_kliep = adaptor_kliep.shift_diagnostic()
print(f"\nRuLSIF verdict: {report_rulsif.verdict}, ESS={report_rulsif.ess_ratio:.3f}")
print(f"KLIEP verdict:  {report_kliep.verdict}, ESS={report_kliep.ess_ratio:.3f}")
