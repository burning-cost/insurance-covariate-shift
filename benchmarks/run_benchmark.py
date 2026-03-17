"""
Benchmark: insurance-covariate-shift
=====================================

Scenario: A motor insurer trained a frequency model on a direct channel book
(younger, urban drivers). The model is then deployed on an acquired broker
portfolio (older, rural drivers). We measure:

1. How well the shift is detected (ESS ratio, verdict)
2. Whether importance-weighted evaluation corrects the biased metric estimate
3. Whether shift-robust conformal intervals achieve the target coverage on the
   target distribution (vs standard conformal which does not)

Baseline: apply model to target data without any correction (naive deployment)
Library:  CovariateShiftAdaptor (rulsif) + ShiftRobustConformal

Seed: 42. All data generated from known DGP.
"""

import time
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from insurance_covariate_shift import CovariateShiftAdaptor, ShiftRobustConformal

# ---------------------------------------------------------------------------
# Data generating process
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

# Source book: direct channel — younger, urban (age mean 35, postcode score mean 10)
n_source = 5_000
X_source = np.column_stack([
    rng.normal(35, 8, n_source),    # driver age
    rng.normal(5,  3, n_source),    # NCD years
    rng.normal(4,  2, n_source),    # vehicle age
    rng.normal(10, 5, n_source),    # postcode risk score (higher = urban)
])
X_source[:, 0] = np.clip(X_source[:, 0], 17, 85)
X_source[:, 1] = np.clip(X_source[:, 1], 0, 9)
X_source[:, 2] = np.clip(X_source[:, 2], 0, 20)
X_source[:, 3] = np.clip(X_source[:, 3], 0, 30)

# True Poisson rate: age-dependent with urban loading
true_log_rate_source = (
    -2.8
    + 0.4 * (X_source[:, 0] < 25).astype(float)  # young driver
    + 0.15 * (X_source[:, 0] >= 70).astype(float) # older driver
    - 0.08 * X_source[:, 1]                        # NCD protection
    + 0.02 * X_source[:, 2]                        # vehicle age
    + 0.015 * X_source[:, 3]                       # urban loading
)
y_source = rng.poisson(np.exp(true_log_rate_source))

# Target book: acquired broker — older, rural (age mean 48, postcode score mean 4)
n_target = 3_000
X_target = np.column_stack([
    rng.normal(48, 10, n_target),
    rng.normal(7,  2,  n_target),
    rng.normal(6,  3,  n_target),
    rng.normal(4,  3,  n_target),
])
X_target[:, 0] = np.clip(X_target[:, 0], 17, 85)
X_target[:, 1] = np.clip(X_target[:, 1], 0, 9)
X_target[:, 2] = np.clip(X_target[:, 2], 0, 20)
X_target[:, 3] = np.clip(X_target[:, 3], 0, 30)

true_log_rate_target = (
    -2.8
    + 0.4 * (X_target[:, 0] < 25).astype(float)
    + 0.15 * (X_target[:, 0] >= 70).astype(float)
    - 0.08 * X_target[:, 1]
    + 0.02 * X_target[:, 2]
    + 0.015 * X_target[:, 3]
)
y_target = rng.poisson(np.exp(true_log_rate_target))

# Split source into train / calibration
n_train = 4_000
X_train, y_train = X_source[:n_train], y_source[:n_train]
X_cal,   y_cal   = X_source[n_train:], y_source[n_train:]

print("=" * 60)
print("Benchmark: insurance-covariate-shift")
print("=" * 60)
print(f"\nSource book (direct): n={n_source}, mean driver age={X_source[:,0].mean():.1f}")
print(f"Target book (broker): n={n_target}, mean driver age={X_target[:,0].mean():.1f}")
print(f"Source mean frequency: {y_source.mean():.4f}")
print(f"Target mean frequency: {y_target.mean():.4f}")

# ---------------------------------------------------------------------------
# Step 1: Train frequency model on source book
# ---------------------------------------------------------------------------
t0 = time.time()
model = PoissonRegressor(alpha=0.01, max_iter=500)
model.fit(X_train, y_train)
fit_time = time.time() - t0
print(f"\nPoisson GLM fit time: {fit_time:.2f}s")

# ---------------------------------------------------------------------------
# Step 2: Baseline — naive metric on source calibration set
# ---------------------------------------------------------------------------
y_pred_cal = model.predict(X_cal)
y_pred_target = model.predict(X_target)

# True target-book deviance (oracle: we know target labels)
deviance_target_true = mean_poisson_deviance(y_target, y_pred_target)

# Naive metric: just compute deviance on source calibration set
deviance_source_naive = mean_poisson_deviance(y_cal, y_pred_cal)

print("\n--- Baseline (naive deployment without correction) ---")
print(f"  Source-cal deviance:       {deviance_source_naive:.6f}")
print(f"  Target deviance (oracle):  {deviance_target_true:.6f}")
print(f"  Bias (source - target):   {deviance_source_naive - deviance_target_true:+.6f}")

# Naive A/E on source cal: will this reflect target performance?
ae_source = y_cal.mean() / y_pred_cal.mean()
ae_target = y_target.mean() / y_pred_target.mean()
print(f"  Source-cal A/E:            {ae_source:.4f}")
print(f"  Target A/E (oracle):       {ae_target:.4f}")
print(f"  A/E bias:                 {ae_source - ae_target:+.4f}")

# ---------------------------------------------------------------------------
# Step 3: Fit CovariateShiftAdaptor (rulsif)
# ---------------------------------------------------------------------------
t0 = time.time()
adaptor = CovariateShiftAdaptor(
    method="rulsif",
    clip_quantile=0.99,
)
adaptor.fit(X_source, X_target, feature_names=["driver_age", "ncd_years", "vehicle_age", "postcode_score"])
adaptor_time = time.time() - t0
print(f"\n--- Shift detection (RuLSIF) ---")
print(f"  Adaptor fit time:          {adaptor_time:.2f}s")

report = adaptor.shift_diagnostic(source_label="Direct", target_label="Broker")
print(f"  ESS ratio:                 {report.ess_ratio:.3f}")
print(f"  KL divergence:             {report.kl_divergence:.3f} nats")
print(f"  Verdict:                   {report.verdict}")

# ---------------------------------------------------------------------------
# Step 4: Importance-weighted metric on source calibration set
# ---------------------------------------------------------------------------
weights_cal = adaptor.importance_weights(X_cal)
weights_cal_norm = weights_cal / weights_cal.mean()

residuals_cal = np.abs(y_cal - y_pred_cal)
mae_naive   = np.mean(residuals_cal)
mae_weighted = np.average(residuals_cal, weights=weights_cal_norm)
mae_target   = np.mean(np.abs(y_target - y_pred_target))

print("\n--- Importance-weighted metric correction ---")
print(f"  Naive MAE on source cal:           {mae_naive:.4f}")
print(f"  Importance-weighted MAE estimate:  {mae_weighted:.4f}")
print(f"  True target MAE (oracle):          {mae_target:.4f}")
print(f"  Naive error vs oracle:    {abs(mae_naive - mae_target):.4f}")
print(f"  Weighted error vs oracle: {abs(mae_weighted - mae_target):.4f}")

improvement = (abs(mae_naive - mae_target) - abs(mae_weighted - mae_target)) / abs(mae_naive - mae_target) * 100
print(f"  Metric correction improvement:     {improvement:.1f}%")

# ---------------------------------------------------------------------------
# Step 5: Conformal prediction intervals — standard vs shift-robust
# ---------------------------------------------------------------------------
alpha = 0.10  # 90% coverage target

# Standard conformal: calibrate on source, apply to target (naive)
# Calibrate using source calibration residuals as nonconformity scores
residuals_abs = np.abs(y_cal - y_pred_cal)
q_standard = float(np.quantile(residuals_abs, 1 - alpha))

lower_std = y_pred_target - q_standard
upper_std = y_pred_target + q_standard
coverage_standard = float(np.mean((y_target >= lower_std) & (y_target <= upper_std)))
width_standard = float(np.mean(upper_std - lower_std))

t0 = time.time()
cp = ShiftRobustConformal(model=model, adaptor=adaptor, alpha=alpha, method="weighted")
cp.calibrate(X_cal, y_cal)
lower_robust, upper_robust = cp.predict_interval(X_target)
robust_time = time.time() - t0

coverage_robust = float(np.mean((y_target >= lower_robust) & (y_target <= upper_robust)))
width_robust = float(np.mean(upper_robust - lower_robust))

print(f"\n--- Conformal intervals (target = 90% coverage) ---")
print(f"  Standard conformal:  coverage={coverage_standard:.3f}  width={width_standard:.4f}")
print(f"  Shift-robust:        coverage={coverage_robust:.3f}  width={width_robust:.4f}")
print(f"  Shift-robust fit time: {robust_time:.2f}s")
print(f"  Standard undercoverage: {max(0, 0.9 - coverage_standard):.3f}")
print(f"  Robust undercoverage:   {max(0, 0.9 - coverage_robust):.3f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Shift detected: {report.verdict} (ESS={report.ess_ratio:.3f})")
print(f"  Metric correction: naive error {abs(mae_naive - mae_target):.4f} → "
      f"weighted {abs(mae_weighted - mae_target):.4f} ({improvement:.0f}% improvement)")
print(f"  Standard conformal coverage on target: {coverage_standard:.3f} (target: 0.900)")
print(f"  Shift-robust coverage on target:       {coverage_robust:.3f} (target: 0.900)")
