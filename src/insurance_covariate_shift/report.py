"""
ShiftDiagnosticReport — pricing governance documentation of distribution shift.

The report is the artefact that goes to the pricing governance committee and,
in severe cases, into the regulatory filing. It should answer three questions:
1. How big is the shift?
2. Which features drive it?
3. What is the recommended action?
"""

from __future__ import annotations

import textwrap
import warnings
from datetime import date
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ._types import CovariateShiftConfig, ShiftVerdict

__all__ = ["ShiftDiagnosticReport"]


class ShiftDiagnosticReport:
    """
    Summary of the covariate shift between a source and target book.

    This class is returned by :meth:`CovariateShiftAdaptor.shift_diagnostic`.
    You can also construct one directly if you have weights from an external
    process.

    Parameters
    ----------
    weights : array of shape (n_source,)
        Importance weights w(x) = p_target(x) / p_source(x) for each
        source observation.
    feature_names : sequence of str
        Column names corresponding to the feature matrix used to fit the
        adaptor. Used in ``plot_feature_shifts`` and ``feature_importance``.
    feature_importances : array of shape (n_features,) or None
        Per-feature classifier importance from the density ratio model.
        None if not available (e.g. RuLSIF).
    X_source : array of shape (n_source, d) or None
        Source feature matrix — used for distribution plots.
    X_target : array of shape (n_target, d) or None
        Target feature matrix — used for distribution plots.
    config : CovariateShiftConfig
        Thresholds for NEGLIGIBLE / MODERATE / SEVERE classification.
    source_label : str
        Label for the source book in plots and reports. Default "Source".
    target_label : str
        Label for the target book in reports. Default "Target".
    report_date : date or None
        Date to embed in regulatory summary. Defaults to today.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_covariate_shift.report import ShiftDiagnosticReport
    >>> rng = np.random.default_rng(0)
    >>> weights = rng.lognormal(0, 0.3, 400)
    >>> report = ShiftDiagnosticReport(weights=weights, feature_names=["age", "ncb"])
    >>> report.verdict
    'NEGLIGIBLE'
    >>> print(report.fca_sup153_summary())  # doctest: +ELLIPSIS
    Distribution Shift Assessment...
    """

    def __init__(
        self,
        weights: NDArray[np.float64],
        feature_names: Sequence[str] = (),
        feature_importances: Optional[NDArray[np.float64]] = None,
        X_source: Optional[NDArray[np.float64]] = None,
        X_target: Optional[NDArray[np.float64]] = None,
        config: Optional[CovariateShiftConfig] = None,
        source_label: str = "Source",
        target_label: str = "Target",
        report_date: Optional[date] = None,
    ) -> None:
        self._weights = np.asarray(weights, dtype=float)
        self._feature_names = list(feature_names)
        self._feature_importances = (
            np.asarray(feature_importances, dtype=float)
            if feature_importances is not None
            else None
        )
        self._X_source = X_source
        self._X_target = X_target
        self._config = config or CovariateShiftConfig()
        self.source_label = source_label
        self.target_label = target_label
        self._report_date = report_date or date.today()

        # Compute summary statistics
        self.ess_ratio: float = self._compute_ess_ratio()
        self.kl_divergence: float = self._estimate_kl()
        self.verdict: ShiftVerdict = self._config.verdict(self.ess_ratio, self.kl_divergence)

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def _compute_ess_ratio(self) -> float:
        """
        ESS = (sum w_i)^2 / sum w_i^2, normalised by n_source.

        A ratio of 1.0 means the reweighting costs nothing — source and
        target are identical. A ratio of 0.1 means 90% of your source
        data is effectively discarded during adaptation.
        """
        w = self._weights
        n = len(w)
        if n == 0:
            return 0.0
        denom = float(np.sum(w ** 2))
        if denom < 1e-15:
            return 0.0
        ess = float(np.sum(w) ** 2) / denom
        return ess / n

    def _estimate_kl(self) -> float:
        """
        Estimate KL(target || source) from the weights.

        Uses the identity: KL(p_t || p_s) = E_s[w(x) * log w(x)]
        where w(x) = p_t(x)/p_s(x). The identity requires E_s[w] = 1
        (normalised weights). We normalise before computing to ensure
        validity; a UserWarning is issued if the raw weights deviate
        substantially from normalisation.
        """
        w = self._weights
        if len(w) == 0:
            return 0.0
        w_mean = float(w.mean())
        if abs(w_mean - 1.0) > 0.1:
            warnings.warn(
                f"KL divergence estimate: weights have mean {w_mean:.3f} (expected ~1.0). "
                "The KL identity requires E_source[w] = 1; normalising before computing. "
                "This can happen with RuLSIF or CatBoost weights which are not guaranteed "
                "to be normalised. The KL estimate may be less accurate.",
                UserWarning,
                stacklevel=3,
            )
        # Normalise to satisfy E_s[w] = 1 before applying the identity
        w_norm = w / (w_mean + 1e-300)
        kl = float(np.mean(w_norm * np.log(np.clip(w_norm, 1e-10, None))))
        return max(kl, 0.0)

    # ------------------------------------------------------------------
    # Feature-level attribution
    # ------------------------------------------------------------------

    def feature_importance(self) -> dict[str, float]:
        """
        Per-feature contribution to the shift, as a normalised dict.

        When the CatBoost classifier method is used, these are the
        classifier's feature importances — features with high importance
        are the main drivers of the distribution difference.

        Returns
        -------
        dict mapping feature name to importance score (sums to 1.0).
        Returns an empty dict if no importances are available.
        """
        if self._feature_importances is None or len(self._feature_importances) == 0:
            return {}
        names = self._feature_names or [f"f{i}" for i in range(len(self._feature_importances))]
        total = float(self._feature_importances.sum())
        if total < 1e-12:
            return {n: 0.0 for n in names}
        scores = self._feature_importances / total
        return dict(zip(names, scores.tolist()))

    # ------------------------------------------------------------------
    # Regulatory summary
    # ------------------------------------------------------------------

    def fca_sup153_summary(self) -> str:
        """
        Generate a plain-text summary suitable for inclusion in pricing
        governance documentation under PS21/5 and Consumer Duty FG22/5.

        Note: the method name retains the legacy 'fca_sup153_summary' name
        for API compatibility. The correct regulatory references are PS21/5
        (General Insurance Pricing Practices) and FG22/5 (Consumer Duty).
        SUP 15.3 covers material change notifications and is not the primary
        reference for pricing fairness documentation. The actual notification
        trigger is the materiality threshold in the firm's own model change
        policy, not a fixed regulatory rule.

        The output follows the structure recommended in the FCA's pricing
        practices guidance: state the issue, quantify it, and describe the
        action taken.

        Returns
        -------
        str
        """
        fi = self.feature_importance()
        top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = (
            ", ".join(f"{n} ({v:.1%})" for n, v in top_features)
            if top_features
            else "not available"
        )

        action_map: dict[ShiftVerdict, str] = {
            "NEGLIGIBLE": (
                "No adaptation is required. The source model may be applied "
                "to the target book without reweighting."
            ),
            "MODERATE": (
                "Importance weighting is recommended before deploying the source "
                "model on the target book. Monitor weighted loss metrics after "
                "deployment for at least three months."
            ),
            "SEVERE": (
                "Retraining on the target book (or a pooled dataset) is strongly "
                "recommended. If retraining is not feasible in the deployment "
                "timeframe, importance weighting MUST be applied and the "
                "monitoring period extended to six months minimum. Consider "
                "notifying the Chief Actuary before deployment."
            ),
        }

        summary = textwrap.dedent(f"""\
            Distribution Shift Assessment
            ==============================
            Date: {self._report_date.isoformat()}
            Source: {self.source_label}
            Target: {self.target_label}

            Verdict: {self.verdict}

            Metrics
            -------
            Effective Sample Size ratio : {self.ess_ratio:.3f}
              (1.0 = no shift, 0.0 = complete overlap failure)
            KL divergence (target || source) : {self.kl_divergence:.2f} nats (approximate)

            Main drivers of shift
            ---------------------
            {top_str}

            Recommended action
            ------------------
            {action_map[self.verdict]}

            Methodology
            -----------
            Density ratio estimated using insurance-covariate-shift v0.1.0.
            ESS ratio = (sum w)^2 / (n * sum w^2). KL estimated via
            E_source[w * log w] with normalised weights. Thresholds: SEVERE
            if ESS < 0.30 or KL > 0.50 nats; MODERATE if ESS < 0.60 or
            KL > 0.10 nats.
        """)
        return summary

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_weight_distribution(
        self,
        ax=None,
        bins: int = 50,
        title: str = "Importance Weight Distribution",
    ):
        """
        Histogram of the importance weights with ESS annotation.

        A well-behaved reweighting should show weights concentrated near
        1.0 with modest spread. Very heavy tails indicate that a small
        fraction of the source data carries most of the adjustment — a
        warning sign.

        Parameters
        ----------
        ax : matplotlib Axes or None
            If None, a new figure is created.
        bins : int
            Number of histogram bins.
        title : str
            Plot title.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.hist(self._weights, bins=bins, edgecolor="white", color="#1f77b4", alpha=0.85)
        ax.axvline(1.0, color="crimson", linewidth=1.5, linestyle="--", label="w = 1")
        ax.set_xlabel("Importance weight w(x)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.annotate(
            f"ESS ratio = {self.ess_ratio:.3f}\nVerdict: {self.verdict}",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"),
        )
        return ax

    def plot_feature_shifts(
        self,
        features: Optional[Sequence[str]] = None,
        n_cols: int = 3,
        figsize_per_panel: tuple[float, float] = (4.0, 3.0),
    ):
        """
        Side-by-side marginal distributions (source vs target) for each feature.

        Categorical features are shown as bar charts; continuous features as
        overlapping histograms. Only available when X_source and X_target
        were provided at construction.

        Parameters
        ----------
        features : sequence of str or None
            Subset of features to plot. None plots all.
        n_cols : int
            Number of columns in the subplot grid.
        figsize_per_panel : tuple
            (width, height) per subplot panel.

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if self._X_source is None or self._X_target is None:
            raise ValueError("X_source and X_target must be provided at construction for feature plots.")

        names = self._feature_names or [f"f{i}" for i in range(self._X_source.shape[1])]
        if features is not None:
            indices = [names.index(f) for f in features]
            names = [names[i] for i in indices]
        else:
            indices = list(range(len(names)))

        n = len(names)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * figsize_per_panel[0], n_rows * figsize_per_panel[1]),
        )
        axes_flat = np.asarray(axes).flatten()

        for plot_idx, (feat_idx, feat_name) in enumerate(zip(indices, names)):
            ax = axes_flat[plot_idx]
            xs = self._X_source[:, feat_idx]
            xt = self._X_target[:, feat_idx]

            # Detect if categorical (low cardinality or object)
            n_unique = len(set(xs.tolist() + xt.tolist()))
            if n_unique <= 20:
                cats = sorted(set(xs.tolist() + xt.tolist()))
                s_counts = np.array([np.sum(xs == c) for c in cats], dtype=float)
                t_counts = np.array([np.sum(xt == c) for c in cats], dtype=float)
                s_counts /= s_counts.sum() + 1e-12
                t_counts /= t_counts.sum() + 1e-12
                x_pos = np.arange(len(cats))
                ax.bar(x_pos - 0.2, s_counts, width=0.4, label=self.source_label, alpha=0.7, color="#1f77b4")
                ax.bar(x_pos + 0.2, t_counts, width=0.4, label=self.target_label, alpha=0.7, color="#ff7f0e")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(c) for c in cats], rotation=45, fontsize=7)
            else:
                ax.hist(xs, bins=30, density=True, alpha=0.5, label=self.source_label, color="#1f77b4")
                ax.hist(xt, bins=30, density=True, alpha=0.5, label=self.target_label, color="#ff7f0e")

            ax.set_title(feat_name, fontsize=9)
            ax.legend(fontsize=7)

        # Hide unused panels
        for ax in axes_flat[n:]:
            ax.set_visible(False)

        fig.suptitle(
            f"Feature Distributions: {self.source_label} vs {self.target_label}",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"ShiftDiagnosticReport("
            f"verdict={self.verdict!r}, "
            f"ess_ratio={self.ess_ratio:.3f}, "
            f"kl_divergence={self.kl_divergence:.4f}, "
            f"n_weights={len(self._weights)})"
        )
