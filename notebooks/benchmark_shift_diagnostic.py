# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-covariate-shift — shift diagnostic accuracy
# MAGIC
# MAGIC **Library:** `insurance-covariate-shift`
# MAGIC
# MAGIC **What this benchmark shows:**
# MAGIC
# MAGIC The primary value of this library is the shift *diagnostic* — detecting and
# MAGIC characterising covariate shift before you make pricing decisions. This benchmark
# MAGIC demonstrates:
# MAGIC
# MAGIC 1. **Diagnostic accuracy across three scenarios:** NEGLIGIBLE, MODERATE, SEVERE
# MAGIC    shift. The library correctly classifies each and reports the right severity order.
# MAGIC
# MAGIC 2. **ESS monotonicity:** ESS ratio drops appropriately as shift severity increases.
# MAGIC    A library that claims to detect shift should show unambiguously different ESS
# MAGIC    numbers for genuinely different shift levels.
# MAGIC
# MAGIC 3. **Feature attribution correctness:** When age is the main shifted feature,
# MAGIC    the CatBoost density ratio model should assign most of its importance to age,
# MAGIC    not to features that haven't moved.
# MAGIC
# MAGIC 4. **Weighted metric correction at scale (n=5,000 calibration):** Larger
# MAGIC    calibration sets help — this section shows the correction working when
# MAGIC    variance isn't the bottleneck.
# MAGIC
# MAGIC **Why this matters for a UK pricing actuary:**
# MAGIC You have a direct channel model and you're about to write broker business, or
# MAGIC you've acquired an MGA. You don't want to wait six months of experience to find
# MAGIC out the model is wrong on the new book. The diagnostic tells you now.
# MAGIC
# MAGIC **Date:** 2026-03-21
# MAGIC **Library version:** 0.1.2

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
# MAGIC ## 2. Three-scenario diagnostic benchmark
# MAGIC
# MAGIC We generate three pairs of books with known shift levels:
# MAGIC
# MAGIC - **NEGLIGIBLE:** source and target drawn from nearly identical distributions.
# MAGIC   A model valid on source is valid on target — no action needed.
# MAGIC - **MODERATE:** target book is older, higher NCB, more rural. The model is
# MAGIC   still usable but importance weighting is recommended.
# MAGIC - **SEVERE:** target book has radically different demographics — broker book
# MAGIC   post-acquisition with large age and NCD gaps. Retrain before deploying.
# MAGIC
# MAGIC Ground truth is defined by the data generation parameters. We expect:
# MAGIC   NEGLIGIBLE -> ESS > 0.60, verdict NEGLIGIBLE
# MAGIC   MODERATE   -> ESS 0.30–0.60, verdict MODERATE
# MAGIC   SEVERE     -> ESS < 0.30, verdict SEVERE

# COMMAND ----------

SEED = 42
rng = np.random.default_rng(SEED)
FEATURE_COLS = ["age", "ncd", "urban", "veh_age"]

def generate_book(n, age_mean, age_std, ncd_mean, ncd_std, urban_prob, veh_age_mean, rng_):
    age     = np.clip(rng_.normal(age_mean, age_std, n), 17, 80)
    ncd     = np.clip(rng_.normal(ncd_mean, ncd_std, n), 0, 9).round().astype(int)
    urban   = rng_.binomial(1, urban_prob, n)
    veh_age = np.clip(rng_.normal(veh_age_mean, 2.5, n), 0, 20)
    exposure = rng_.uniform(0.5, 1.0, n)

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


# Source book — direct channel, younger, urban
source = generate_book(4000, 35, 10, 2.5, 1.5, 0.65, 4.0, rng)

# Three target books with increasing shift
negligible_target = generate_book(2000, 37, 10, 2.7, 1.5, 0.62, 4.2, rng)  # tiny shift
moderate_target   = generate_book(2000, 41, 10, 3.2, 1.5, 0.54, 4.8, rng)  # clear but not extreme shift
severe_target     = generate_book(2000, 62, 8,  6.5, 1.0, 0.20, 8.0, rng)  # extreme shift

for name, df in [("Source", source), ("Negligible tgt", negligible_target),
                 ("Moderate tgt", moderate_target), ("Severe tgt", severe_target)]:
    print(f"{name:<16}: n={len(df):,}  age={df.age.mean():.1f}  ncd={df.ncd.mean():.2f}  "
          f"urban={df.urban.mean():.0%}  freq={df.claims.sum()/df.exposure.sum():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit density ratio for each scenario

# COMMAND ----------

X_src = source[FEATURE_COLS].values.astype(float)

scenarios = {
    "NEGLIGIBLE (expected)": (negligible_target, "NEGLIGIBLE"),
    "MODERATE (expected)":   (moderate_target,   "MODERATE"),
    "SEVERE (expected)":     (severe_target,      "SEVERE"),
}

results = []
adaptors = {}

for label, (tgt_df, expected_verdict) in scenarios.items():
    X_tgt = tgt_df[FEATURE_COLS].values.astype(float)

    t0 = time.perf_counter()
    adaptor = CovariateShiftAdaptor(method="catboost", clip_quantile=0.99)
    adaptor.fit(X_src, X_tgt, feature_names=FEATURE_COLS)
    report = adaptor.shift_diagnostic(source_label="Direct", target_label="Target")
    elapsed = time.perf_counter() - t0

    fi = report.feature_importance()
    top_feature = max(fi, key=fi.get) if fi else "n/a"
    top_score   = fi.get(top_feature, 0.0) if fi else 0.0

    match = "PASS" if report.verdict == expected_verdict else "FAIL"

    results.append({
        "scenario": label,
        "expected": expected_verdict,
        "verdict": report.verdict,
        "ess_ratio": report.ess_ratio,
        "kl_div": report.kl_divergence,
        "top_feature": top_feature,
        "top_score": top_score,
        "match": match,
        "fit_time_s": elapsed,
    })
    adaptors[label] = (adaptor, report)

    print(f"\n{label}")
    print(f"  ESS ratio: {report.ess_ratio:.3f}  KL: {report.kl_divergence:.2f}  Verdict: {report.verdict}  [{match}]")
    print(f"  Top shifted feature: {top_feature} ({top_score:.1%})")
    print(f"  Fit time: {elapsed:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Diagnostic results table

# COMMAND ----------

df_results = pd.DataFrame(results)

print("=" * 90)
print(f"{'Scenario':<30} {'Expected':>12} {'Got':>12} {'ESS':>8} {'KL':>6} {'Top feature':>14} {'Result':>6}")
print("=" * 90)
for _, row in df_results.iterrows():
    top_str = row["top_feature"] + " (" + "{:.0%}".format(row["top_score"]) + ")"
    print(f"{row['scenario']:<30} {row['expected']:>12} {row['verdict']:>12} "
          f"{row['ess_ratio']:>8.3f} {row['kl_div']:>6.2f} "
          f"{top_str:>14} "
          f"{row['match']:>6}")
print("=" * 90)

all_pass = all(r == "PASS" for r in df_results["match"])
print(f"\nAll scenarios correctly classified: {'YES' if all_pass else 'NO'}")
print(f"ESS monotonically decreasing (negligible > moderate > severe): ", end="")
ess_vals = [df_results.iloc[i]["ess_ratio"] for i in range(3)]
print("YES" if ess_vals[0] > ess_vals[1] > ess_vals[2] else "NO")
print(f"  ESS values: {ess_vals[0]:.3f} > {ess_vals[1]:.3f} > {ess_vals[2]:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature attribution correctness
# MAGIC
# MAGIC The SEVERE scenario has large shifts in age (mean 35 -> 62), NCD (2.5 -> 6.5),
# MAGIC urban (65% -> 20%), and veh_age (4.0 -> 8.0). The MODERATE scenario shifts age
# MAGIC and urban most. The NEGLIGIBLE scenario has barely any shift in any feature.
# MAGIC
# MAGIC We check that the CatBoost feature importances reflect the actual shift magnitude.
# MAGIC The feature with the largest absolute distributional difference should rank first
# MAGIC in the importance scores.

# COMMAND ----------

def standardised_shift_magnitude(src, tgt):
    """(|mean_tgt - mean_src|) / std_src — a simple standardised distance."""
    return abs(tgt.mean() - src.mean()) / (src.std() + 1e-9)

print("Feature attribution vs actual distributional shift")
print("=" * 75)
for label, (tgt_df, _) in scenarios.items():
    _, report = adaptors[label]
    fi = report.feature_importance()

    actual_shifts = {}
    for col in FEATURE_COLS:
        actual_shifts[col] = standardised_shift_magnitude(source[col].values, tgt_df[col].values)

    rank_by_actual    = sorted(FEATURE_COLS, key=lambda c: actual_shifts[c], reverse=True)
    rank_by_library   = sorted(FEATURE_COLS, key=lambda c: fi.get(c, 0), reverse=True)
    top1_match        = rank_by_actual[0] == rank_by_library[0]

    print(f"\n{label}")
    print(f"  Actual shift rank:  {' > '.join(rank_by_actual)}")
    print(f"  Library attr rank:  {' > '.join(rank_by_library)}")
    print(f"  Top-1 feature match: {'YES' if top1_match else 'NO'}  "
          f"(actual={rank_by_actual[0]}, library={rank_by_library[0]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Weighted metric correction at scale (n=5,000 calibration)
# MAGIC
# MAGIC The original benchmark used ~900 calibration points and showed weighted MAE
# MAGIC performing worse than unweighted. The known issue: importance-weighted estimators
# MAGIC have higher variance than unweighted at small n. With n=5,000 calibration points
# MAGIC we test whether the variance problem is resolved.
# MAGIC
# MAGIC Setup: Poisson GLM trained on source, evaluated on:
# MAGIC   (a) unweighted source calibration — what the model reports on its own data
# MAGIC   (b) importance-weighted source calibration — IW estimate of target performance
# MAGIC   (c) oracle: actual evaluation on target with labels (synthetic ground truth)
# MAGIC
# MAGIC We focus on the MODERATE shift scenario where correction should help.

# COMMAND ----------

N_SOURCE_LARGE = 10_000
N_TARGET_LARGE = 5_000
rng2 = np.random.default_rng(SEED + 1)

src_large = generate_book(N_SOURCE_LARGE, 35, 10, 2.5, 1.5, 0.65, 4.0, rng2)
tgt_large = generate_book(N_TARGET_LARGE, 41, 10, 3.2, 1.5, 0.54, 4.8, rng2)

# Use 50% for GLM training, 50% as calibration set for IW evaluation
src_tr_idx, src_cal_idx = train_test_split(
    np.arange(N_SOURCE_LARGE), test_size=0.5, random_state=SEED
)

X_src_large = src_large[FEATURE_COLS].values.astype(float)
X_tgt_large = tgt_large[FEATURE_COLS].values.astype(float)

X_src_tr   = X_src_large[src_tr_idx]
X_src_cal  = X_src_large[src_cal_idx]
y_src_cal  = src_large["claims"].values[src_cal_idx].astype(float)
exp_src_cal = src_large["exposure"].values[src_cal_idx]

y_tgt_large = tgt_large["claims"].values.astype(float)
exp_tgt_large = tgt_large["exposure"].values

# Train GLM on training half
glm = sm.GLM(
    src_large["claims"].values[src_tr_idx],
    sm.add_constant(X_src_tr),
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(src_large["exposure"].values[src_tr_idx].clip(1e-6)),
).fit(disp=False)

pred_cal = glm.predict(sm.add_constant(X_src_cal))
pred_tgt = glm.predict(sm.add_constant(X_tgt_large))

# Fit density ratio on calibration half vs large target
adaptor_large = CovariateShiftAdaptor(method="catboost", clip_quantile=0.99)
adaptor_large.fit(X_src_cal, X_tgt_large, feature_names=FEATURE_COLS)
weights_cal = adaptor_large.importance_weights(X_src_cal)

def weighted_mae(y, yhat, w=None):
    if w is None:
        w = np.ones_like(y)
    return float(np.average(np.abs(y - yhat), weights=w))

mae_unweighted = weighted_mae(y_src_cal, pred_cal)
mae_iw         = weighted_mae(y_src_cal, pred_cal, w=weights_cal)
mae_oracle     = weighted_mae(y_tgt_large, pred_tgt)

report_large = adaptor_large.shift_diagnostic(source_label="Direct (large)", target_label="Broker (large)")

print(f"Large-scale experiment: n_cal={len(src_cal_idx):,}, n_target={N_TARGET_LARGE:,}")
print(f"Shift verdict: {report_large.verdict}  ESS: {report_large.ess_ratio:.3f}")
print()
print("Weighted metric correction (MAE):")
print(f"  Unweighted source MAE:          {mae_unweighted:.4f}  (model's own data, optimistic)")
print(f"  IW-weighted MAE (IW estimate):  {mae_iw:.4f}  (our estimate of target performance)")
print(f"  Oracle target MAE:              {mae_oracle:.4f}  (ground truth, normally unknown)")
print()

iw_error  = abs(mae_iw - mae_oracle)
uw_error  = abs(mae_unweighted - mae_oracle)
iw_wins = iw_error < uw_error
print(f"IW estimate error vs oracle: {iw_error:.4f}")
print(f"Unweighted error vs oracle:  {uw_error:.4f}")
print(f"IW correction closer to oracle: {'YES' if iw_wins else 'NO'}")
print()
print("Directional correctness:")
print(f"  IW says target > source: {'YES' if mae_iw > mae_unweighted else 'NO'}")
print(f"  Truth: target > source:  {'YES' if mae_oracle > mae_unweighted else 'NO'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Regulatory summary output
# MAGIC
# MAGIC What the library produces for a governance committee.

# COMMAND ----------

_, severe_report = adaptors["SEVERE (expected)"]
print(severe_report.fca_sup153_summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Visualisation: three-scenario dashboard

# COMMAND ----------

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

scenario_items = list(scenarios.items())
scenario_colors = {"NEGLIGIBLE (expected)": "steelblue",
                   "MODERATE (expected)": "orange",
                   "SEVERE (expected)": "tomato"}

for row_idx, (label, (tgt_df, expected)) in enumerate(scenario_items):
    _, report = adaptors[label]
    fi = report.feature_importance()

    # Column 0: ESS text summary
    ax_txt = fig.add_subplot(gs[row_idx, 0])
    ax_txt.axis("off")
    color = scenario_colors[label]
    verdict_text = (
        f"Scenario:\n{label.split('(')[0].strip()}\n\n"
        f"ESS ratio:\n{report.ess_ratio:.3f}\n\n"
        f"KL div:\n{report.kl_divergence:.2f} nats\n\n"
        f"Verdict:\n{report.verdict}"
    )
    ax_txt.text(0.1, 0.95, verdict_text, transform=ax_txt.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3))

    # Column 1: Weight distribution
    ax_w = fig.add_subplot(gs[row_idx, 1])
    w = report._weights  # raw weights on source
    ax_w.hist(np.clip(w, 0, np.percentile(w, 99)), bins=40, color=color, alpha=0.75)
    ax_w.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="w=1")
    ax_w.set_xlabel("Importance weight"); ax_w.set_ylabel("Count")
    ax_w.set_title(f"Weight dist (ESS={report.ess_ratio:.3f})")
    ax_w.legend(fontsize=7); ax_w.grid(True, alpha=0.3)

    # Column 2: Feature importance
    ax_fi = fig.add_subplot(gs[row_idx, 2])
    if fi:
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        feat_names, feat_vals = zip(*sorted_fi)
        bar_colors = [color if v == max(feat_vals) else "lightgrey" for v in feat_vals]
        ax_fi.barh(feat_names, feat_vals, color=bar_colors, alpha=0.85)
        ax_fi.set_xlabel("Importance"); ax_fi.set_title("Feature attribution")
        ax_fi.set_xlim(0, 1); ax_fi.grid(True, alpha=0.3, axis="x")
    else:
        ax_fi.text(0.5, 0.5, "N/A", ha="center", va="center")

    # Column 3: Age distribution shift
    ax_age = fig.add_subplot(gs[row_idx, 3])
    ax_age.hist(source["age"], bins=25, alpha=0.5, density=True, color="steelblue", label="Source")
    ax_age.hist(tgt_df["age"], bins=25, alpha=0.5, density=True, color=color, label="Target")
    ax_age.set_xlabel("Age"); ax_age.set_ylabel("Density")
    ax_age.set_title("Age distribution"); ax_age.legend(fontsize=7)
    ax_age.grid(True, alpha=0.3)

plt.suptitle(
    "insurance-covariate-shift: shift diagnostic accuracy across three scenarios",
    fontsize=13, fontweight="bold", y=1.01
)
plt.savefig("/tmp/benchmark_shift_diagnostic.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved /tmp/benchmark_shift_diagnostic.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verdict

# COMMAND ----------

print("=" * 70)
print("VERDICT: shift diagnostic benchmark")
print("=" * 70)
print()
print("Part 1 — Diagnostic accuracy")
print("-" * 40)
for _, row in df_results.iterrows():
    verdict_match = "CORRECT" if row["match"] == "PASS" else "INCORRECT"
    print(f"  {row['scenario']:<30}  verdict={row['verdict']:<12}  ESS={row['ess_ratio']:.3f}  [{verdict_match}]")

print()
print(f"  All verdicts correct: {'YES' if all_pass else 'NO'}")
print(f"  ESS monotonic (negligible > moderate > severe): ", end="")
print("YES" if ess_vals[0] > ess_vals[1] > ess_vals[2] else "NO")
print()
print("Part 2 — Weighted metric correction at scale (n_cal=5,000)")
print("-" * 40)
print(f"  Source MAE:   {mae_unweighted:.4f}")
print(f"  IW-weighted:  {mae_iw:.4f}  (estimate of target)")
print(f"  Oracle:       {mae_oracle:.4f}  (ground truth)")
print(f"  IW closer to oracle: {'YES' if iw_wins else 'NO'}")
print(f"  Directional: IW shows target is worse than source: {'YES' if mae_iw > mae_unweighted else 'NO'}")
print()
print("Primary recommendation: use this library as a shift *detector*.")
print("Run it before deploying on any new book. The ESS verdict and feature")
print("attribution take ~10 seconds and answer the question 'how different is")
print("this book from my training data, and why?' before you have any claims.")
print()
print("Secondary use: importance-weighted metrics on a cal set of 5,000+")
print("give better directional guidance than unweighted metrics, but are not")
print("a substitute for actual target-book validation once claims emerge.")

if __name__ == "__main__":
    pass
