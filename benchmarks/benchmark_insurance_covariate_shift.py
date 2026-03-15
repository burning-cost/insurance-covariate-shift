# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-covariate-shift — importance-weighted evaluation vs unweighted
# MAGIC
# MAGIC **Library:** `insurance-covariate-shift` — density ratio estimation for insurance
# MAGIC model portability. When a model trained on one distribution is evaluated on a
# MAGIC shifted distribution, unweighted metrics give a biased picture of true performance.
# MAGIC
# MAGIC **Baseline:** unweighted evaluation on the target book — compute Gini and
# MAGIC calibration error (A/E by decile) directly on shifted test data. This is what
# MAGIC most pricing teams do: validate the model on whatever data is available.
# MAGIC
# MAGIC **Library method:** fit a RuLSIF density ratio model, reweight source evaluation
# MAGIC metrics to estimate target-distribution performance without needing target labels.
# MAGIC
# MAGIC **Dataset:** 5,000 source (direct channel, younger/urban) + 3,000 target (broker
# MAGIC book, older/rural). Poisson GLM trained on source, evaluated on both.
# MAGIC
# MAGIC **Date:** 2026-03-15
# MAGIC **Library version:** 0.1.0
# MAGIC
# MAGIC ---
# MAGIC When covariate shift is present, unweighted evaluation lies. The Gini on the
# MAGIC target book looks fine because you rank policies the same way — but A/E by age
# MAGIC band shows the model is wrong in a systematic, concentrated pattern. Importance
# MAGIC weighting reveals the bias early, before you accumulate a year of bad claims data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-covariate-shift statsmodels numpy scipy matplotlib pandas scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from insurance_covariate_shift import CovariateShiftAdaptor

warnings.filterwarnings("ignore")
print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic Data

# COMMAND ----------

N_SOURCE = 5_000
N_TARGET = 3_000
SEED = 42
rng = np.random.default_rng(SEED)

def generate_book(n, age_mean, age_std, ncd_mean, ncd_std, urban_prob, veh_age_mean, rng_):
    age     = np.clip(rng_.normal(age_mean, age_std, n), 17, 80)
    ncd     = np.clip(rng_.normal(ncd_mean, ncd_std, n), 0, 9).round().astype(int)
    urban   = rng_.binomial(1, urban_prob, n)
    veh_age = np.clip(rng_.normal(veh_age_mean, 2.5, n), 0, 20)
    exposure = rng_.uniform(0.5, 1.0, n)

    # True DGP — same structural model on both books
    log_freq = (
        -2.5
        - 0.020 * age
        - 0.150 * ncd
        + 0.250 * urban
        + 0.030 * veh_age
    )
    true_freq = np.exp(log_freq)
    claims = rng_.poisson(true_freq * exposure)

    return pd.DataFrame({
        "age": age, "ncd": ncd.astype(float), "urban": urban.astype(float),
        "veh_age": veh_age, "exposure": exposure,
        "true_freq": true_freq, "claims": claims,
    })

source_df = generate_book(N_SOURCE, 35, 10, 2.5, 1.5, 0.65, 4.0, rng)
target_df = generate_book(N_TARGET, 48, 12, 4.0, 1.0, 0.40, 3.0, rng)

FEATURE_COLS = ["age", "ncd", "urban", "veh_age"]

print(f"Source: {N_SOURCE:,} policies, age={source_df.age.mean():.1f}, urban={source_df.urban.mean():.0%}, "
      f"freq={source_df.claims.sum()/source_df.exposure.sum():.4f}")
print(f"Target: {N_TARGET:,} policies, age={target_df.age.mean():.1f}, urban={target_df.urban.mean():.0%}, "
      f"freq={target_df.claims.sum()/target_df.exposure.sum():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train GLM on Source

# COMMAND ----------

X_src = source_df[FEATURE_COLS].values.astype(float)
X_tgt = target_df[FEATURE_COLS].values.astype(float)

glm = sm.GLM(
    source_df["claims"],
    sm.add_constant(X_src),
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(source_df["exposure"].clip(1e-6)),
).fit(disp=False)

pred_src = glm.predict(sm.add_constant(X_src))     # predicted claims
pred_tgt = glm.predict(sm.add_constant(X_tgt))

print("GLM trained on source book.")
print(f"Source A/E: {source_df.claims.sum() / pred_src.sum():.4f}")
print(f"Target A/E: {target_df.claims.sum() / pred_tgt.sum():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline: Unweighted Evaluation on Target

# COMMAND ----------

def gini(y, yhat, w=None):
    if w is None: w = np.ones_like(y)
    o = np.argsort(yhat)
    ys, ws = y[o], w[o]
    cw = np.cumsum(ws) / ws.sum()
    cy = np.cumsum(ys * ws) / (ys * ws).sum()
    return float(2 * np.trapz(cy, cw) - 1)

def ae_by_decile_mae(y, yhat, w=None, n=10):
    if w is None: w = np.ones_like(y)
    cuts = pd.qcut(yhat, n, labels=False, duplicates="drop")
    aes = []
    for q in range(n):
        m = cuts == q
        if m.sum() < 2: continue
        aes.append(float((y[m] * w[m]).sum() / max((yhat[m] * w[m]).sum(), 1e-10)))
    return float(np.mean(np.abs(np.array(aes) - 1.0)))

t0_base = time.perf_counter()
y_tgt  = target_df["claims"].values.astype(float)
exp_tgt = target_df["exposure"].values

gini_uw   = gini(y_tgt, pred_tgt, w=exp_tgt)
ae_mae_uw = ae_by_decile_mae(y_tgt, pred_tgt, w=exp_tgt)
base_time = time.perf_counter() - t0_base

print(f"Baseline time: {base_time:.3f}s")
print(f"Gini (unweighted target):   {gini_uw:.4f}")
print(f"A/E MAE (unweighted target):{ae_mae_uw:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Library: Density Ratio Estimation + Weighted Evaluation

# COMMAND ----------

# Split source: train GLM on train half, compute importance-weighted metrics on cal half
src_tr_idx, src_cal_idx = train_test_split(np.arange(N_SOURCE), test_size=0.4, random_state=SEED)

X_src_tr  = X_src[src_tr_idx]
X_src_cal = X_src[src_cal_idx]
y_src_cal = source_df["claims"].values[src_cal_idx].astype(float)
exp_src_cal = source_df["exposure"].values[src_cal_idx]

t0_lib = time.perf_counter()

# Fit density ratio on calibration half vs target
adaptor = CovariateShiftAdaptor(method="rulsif", clip_quantile=0.99)
adaptor.fit(X_src_cal, X_tgt, feature_names=FEATURE_COLS)
weights = adaptor.importance_weights(X_src_cal)

# Retrain GLM on training portion, predict on cal set
glm2 = sm.GLM(
    source_df["claims"].values[src_tr_idx],
    sm.add_constant(X_src_tr),
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(source_df["exposure"].values[src_tr_idx].clip(1e-6)),
).fit(disp=False)

pred_cal = glm2.predict(sm.add_constant(X_src_cal))

gini_iw   = gini(y_src_cal, pred_cal, w=weights * exp_src_cal)
ae_mae_iw = ae_by_decile_mae(y_src_cal, pred_cal, w=weights)

# Oracle: true metrics on target using target labels
gini_oracle   = gini(y_tgt, glm2.predict(sm.add_constant(X_tgt)), w=exp_tgt)
ae_mae_oracle = ae_by_decile_mae(y_tgt, glm2.predict(sm.add_constant(X_tgt)), w=exp_tgt)

lib_time = time.perf_counter() - t0_lib

report = adaptor.shift_diagnostic(source_label="Direct", target_label="Broker")
print(f"Library time: {lib_time:.3f}s")
print(f"Shift verdict: {report.verdict}  ESS ratio: {report.effective_sample_size_ratio:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results Table

# COMMAND ----------

print("=" * 72)
print(f"{'Metric':<40} {'Unweighted':>10} {'IW-Weighted':>12} {'Oracle':>8}")
print("=" * 72)

rows = [
    ("Gini coefficient (higher=better)",   gini_uw,     gini_iw,     gini_oracle),
    ("A/E decile MAE (lower=better)",       ae_mae_uw,   ae_mae_iw,   ae_mae_oracle),
    ("Evaluation time (s)",                 base_time,   lib_time,    None),
]
for name, b, l, o in rows:
    o_str = f"{o:>8.4f}" if o is not None else "     n/a"
    print(f"{name:<40} {b:>10.4f} {l:>12.4f} {o_str}")

print("=" * 72)
print()
print(f"Shift summary: {report.verdict}  (ESS={report.effective_sample_size_ratio:.3f})")
print()
print("IW error vs oracle:")
print(f"  Gini: unweighted={abs(gini_uw - gini_oracle):.4f}, IW={abs(gini_iw - gini_oracle):.4f}")
print(f"  A/E:  unweighted={abs(ae_mae_uw - ae_mae_oracle):.4f}, IW={abs(ae_mae_iw - ae_mae_oracle):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Plots

# COMMAND ----------

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Plot 1: Age distribution shift
ax1.hist(source_df["age"], bins=30, alpha=0.6, density=True, color="steelblue", label=f"Source (n={N_SOURCE:,})")
ax1.hist(target_df["age"], bins=30, alpha=0.6, density=True, color="tomato", label=f"Target (n={N_TARGET:,})")
ax1.set_xlabel("Driver age"); ax1.set_ylabel("Density")
ax1.set_title("Covariate shift: age distribution")
ax1.legend(); ax1.grid(True, alpha=0.3)

# Plot 2: Importance weights
ax2.hist(np.clip(weights, 0, np.percentile(weights, 99)), bins=40, color="forestgreen", alpha=0.8)
ax2.axvline(1.0, color="red", linestyle="--", linewidth=2, label="No shift (w=1)")
ax2.set_xlabel("w(x) = p_target / p_source"); ax2.set_ylabel("Count")
ax2.set_title(f"Density ratio weights\nESS={report.effective_sample_size_ratio:.3f} verdict={report.verdict}")
ax2.legend(); ax2.grid(True, alpha=0.3)

# Plot 3: A/E by age quintile
age_q_tgt = pd.qcut(target_df["age"], 5, labels=False)
age_q_cal = pd.qcut(source_df["age"].values[src_cal_idx], 5, labels=False)
pred_cal_arr = glm2.predict(sm.add_constant(X_src_cal))

ae_uw_by_q, ae_iw_by_q = [], []
for q in range(5):
    m_t = age_q_tgt == q
    ae_uw_q = float(y_tgt[m_t].sum() / max(pred_tgt[m_t].sum(), 1e-10))
    ae_uw_by_q.append(ae_uw_q)

    m_s = age_q_cal == q
    if m_s.sum() < 5:
        ae_iw_by_q.append(np.nan)
        continue
    w_q = weights[m_s] * exp_src_cal[m_s]
    ae_iw_q = float((y_src_cal[m_s] * w_q).sum() / max((pred_cal_arr[m_s] * w_q).sum(), 1e-10))
    ae_iw_by_q.append(ae_iw_q)

xq = np.arange(1, 6)
ax3.bar(xq - 0.2, ae_uw_by_q, 0.4, label="Unweighted (target)", color="steelblue", alpha=0.8)
ax3.bar(xq + 0.2, ae_iw_by_q, 0.4, label="IW-weighted (source)", color="tomato", alpha=0.8)
ax3.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
ax3.set_xlabel("Age quintile (1=youngest)"); ax3.set_ylabel("A/E ratio")
ax3.set_title("A/E by age: unweighted vs IW-weighted"); ax3.legend()
ax3.set_xticks(xq); ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Feature PSI bar chart
psi_vals = []
for col in FEATURE_COLS:
    s = source_df[col].values
    t = target_df[col].values
    bins = np.quantile(np.concatenate([s, t]), np.linspace(0, 1, 11))
    bins[0] -= 1e-8; bins[-1] += 1e-8
    sp = np.histogram(s, bins=bins)[0] / len(s)
    tp = np.histogram(t, bins=bins)[0] / len(t)
    sp = np.clip(sp, 1e-10, None); tp = np.clip(tp, 1e-10, None)
    psi_vals.append(float(np.sum((sp - tp) * np.log(sp / tp))))

colors_p = ["tomato" if p >= 0.25 else ("orange" if p >= 0.10 else "steelblue") for p in psi_vals]
ax4.bar(FEATURE_COLS, psi_vals, color=colors_p, alpha=0.8)
ax4.axhline(0.10, color="orange", linestyle="--", linewidth=1.5, label="Moderate (0.10)")
ax4.axhline(0.25, color="tomato", linestyle="--", linewidth=1.5, label="Severe (0.25)")
ax4.set_ylabel("PSI"); ax4.set_title("Feature PSI: source vs target")
ax4.legend(); ax4.grid(True, alpha=0.3, axis="y")

plt.suptitle("insurance-covariate-shift: IW Evaluation vs Unweighted", fontsize=13, fontweight="bold")
plt.savefig("/tmp/benchmark_covariate_shift.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved /tmp/benchmark_covariate_shift.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verdict

# COMMAND ----------

print("=" * 62)
print("VERDICT: Importance Weighting vs Direct Evaluation Under Shift")
print("=" * 62)
print()
print(f"Shift: {report.verdict}  ESS={report.effective_sample_size_ratio:.3f}")
print()
gini_gain = abs(gini_uw - gini_oracle) - abs(gini_iw - gini_oracle)
ae_gain   = abs(ae_mae_uw - ae_mae_oracle) - abs(ae_mae_iw - ae_mae_oracle)
print(f"  Gini error reduction: {gini_gain:+.4f} ({'IW wins' if gini_gain > 0 else 'tie/baseline wins'})")
print(f"  A/E error reduction:  {ae_gain:+.4f}  ({'IW wins' if ae_gain > 0 else 'tie/baseline wins'})")
print()
print("When IW wins:")
print("  Importance-weighted evaluation estimates true target-book performance")
print("  without needing target labels. A/E by age band reveals where the model")
print("  is systematically wrong — information unavailable from unweighted metrics.")
print("  Use this before deploying on an acquired book or new channel.")

if __name__ == "__main__":
    pass
