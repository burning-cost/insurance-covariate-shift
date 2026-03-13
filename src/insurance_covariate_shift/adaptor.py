"""
CovariateShiftAdaptor — the main interface for density ratio estimation.

The core idea is simple: train a binary classifier to distinguish source from
target observations. If it can do so confidently, the two distributions are
different and the classifier's probabilities tell you by how much. The weight
for a source observation is then:

    w(x) = (n_t / n_s) * P(target | x) / P(source | x)

This is the standard Shimodaira (2000) result. CatBoost is the default
because it handles high-cardinality categoricals natively — postcode,
vehicle make-model, occupation — without any preprocessing, which is the
main practical headache in UK insurance.

RuLSIF and KLIEP are provided as alternatives when features are all
continuous (e.g. model scores, rating factors) and you want closed-form
guarantees rather than relying on a classifier.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from ._types import CovariateShiftConfig, DensityRatioMethod, ShiftVerdict
from .density_ratio import KLIEP, RuLSIF
from .report import ShiftDiagnosticReport

__all__ = ["CovariateShiftAdaptor"]


class CovariateShiftAdaptor:
    """
    Estimate density ratio p_target(x) / p_source(x) from two unlabelled datasets.

    This adaptor sits between your existing model and a new book of business.
    You pass it your source training set and the target deployment set, it
    learns a reweighting function, and you use those weights when evaluating
    or recalibrating your model on the target.

    Parameters
    ----------
    method : {'catboost', 'rulsif', 'kliep'}
        Density ratio estimation method.

        * ``'catboost'`` (default) — trains a binary logistic classifier.
          Handles categorical features natively. Best for UK insurance data
          with postcodes and vehicle codes.
        * ``'rulsif'`` — Relative Unconstrained Least-Squares Importance
          Fitting. Closed-form, fast, but requires all-continuous features.
        * ``'kliep'`` — KL Importance Estimation Procedure. Projected gradient
          ascent with normalisation constraint. Slower but unbiased.

    categorical_cols : list of int or None
        Column indices that are categorical, for the CatBoost method. If None,
        all features are treated as continuous.
    exposure_col : int or None
        Column index of an exposure / weight column in the feature matrix.
        When provided, this column is excluded from the density ratio
        model but its values are used to scale the final weights. This
        handles the common case where source and target have different
        average exposures.
    clip_quantile : float
        Quantile at which to clip extreme weights. Default 0.99.
        Extreme weights cause variance blow-up in downstream estimates.
    catboost_iterations : int
        Number of boosting iterations for the CatBoost classifier.
        Default 300. Reduce if fitting on very small datasets.
    catboost_depth : int
        Tree depth for CatBoost. Default 6.
    rulsif_alpha : float
        Alpha for the RuLSIF relative ratio. Default 0.1.
    config : CovariateShiftConfig or None
        Thresholds for shift verdict classification. None uses defaults.

    Attributes
    ----------
    is_fitted_ : bool
        True after :meth:`fit` has been called.
    clip_threshold_ : float
        Computed weight clip value (``clip_quantile`` percentile of raw weights).

    Examples
    --------
    Fit on a motor book acquisition:

    >>> import numpy as np
    >>> from insurance_covariate_shift import CovariateShiftAdaptor
    >>> rng = np.random.default_rng(42)
    >>> X_source = rng.normal(0, 1, (500, 4))
    >>> X_target = rng.normal(0.5, 1.2, (300, 4))
    >>> adaptor = CovariateShiftAdaptor(method='rulsif')
    >>> adaptor.fit(X_source, X_target)
    CovariateShiftAdaptor(method='rulsif')
    >>> w = adaptor.importance_weights(X_source)
    >>> w.shape
    (500,)
    >>> w.min() >= 0
    True
    """

    def __init__(
        self,
        method: DensityRatioMethod = "catboost",
        categorical_cols: Optional[List[int]] = None,
        exposure_col: Optional[int] = None,
        clip_quantile: float = 0.99,
        catboost_iterations: int = 300,
        catboost_depth: int = 6,
        rulsif_alpha: float = 0.1,
        config: Optional[CovariateShiftConfig] = None,
    ) -> None:
        if method not in ("catboost", "rulsif", "kliep"):
            raise ValueError(f"Unknown method {method!r}. Choose 'catboost', 'rulsif', or 'kliep'.")
        if not (0.5 < clip_quantile <= 1.0):
            raise ValueError(f"clip_quantile must be in (0.5, 1.0], got {clip_quantile}")

        self.method = method
        self.categorical_cols = categorical_cols or []
        self.exposure_col = exposure_col
        self.clip_quantile = clip_quantile
        self.catboost_iterations = catboost_iterations
        self.catboost_depth = catboost_depth
        self.rulsif_alpha = rulsif_alpha
        self.config = config or CovariateShiftConfig()

        self._model = None
        self._n_source: int = 0
        self._n_target: int = 0
        self._feature_cols: List[int] = []
        self._feature_names: List[str] = []
        self.clip_threshold_: float = np.inf
        self.is_fitted_: bool = False

        # Stored for diagnostics
        self._X_source_fit: Optional[NDArray[np.float64]] = None
        self._X_target_fit: Optional[NDArray[np.float64]] = None
        self._raw_weights_source: Optional[NDArray[np.float64]] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X_source: NDArray,
        X_target: NDArray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "CovariateShiftAdaptor":
        """
        Train the density ratio model.

        Parameters
        ----------
        X_source : array-like of shape (n_source, d)
            Feature matrix for the source book (training distribution).
        X_target : array-like of shape (n_target, d)
            Feature matrix for the target book (deployment distribution).
            Labels are not needed — this is unsupervised adaptation.
        feature_names : sequence of str or None
            Column names. Used in diagnostic reports and plots.

        Returns
        -------
        self
        """
        X_source = np.asarray(X_source)
        X_target = np.asarray(X_target)

        if X_source.ndim == 1:
            X_source = X_source.reshape(-1, 1)
        if X_target.ndim == 1:
            X_target = X_target.reshape(-1, 1)

        if X_source.shape[1] != X_target.shape[1]:
            raise ValueError(
                f"X_source has {X_source.shape[1]} columns but X_target has {X_target.shape[1]}."
            )

        n_cols = X_source.shape[1]
        self._feature_cols = [i for i in range(n_cols) if i != self.exposure_col]
        self._n_source = len(X_source)
        self._n_target = len(X_target)
        self._feature_names = list(feature_names) if feature_names else [f"f{i}" for i in self._feature_cols]

        Xs = X_source[:, self._feature_cols]
        Xt = X_target[:, self._feature_cols]

        if self.method == "catboost":
            self._fit_catboost(Xs, Xt)
        elif self.method == "rulsif":
            self._fit_rulsif(Xs, Xt)
        elif self.method == "kliep":
            self._fit_kliep(Xs, Xt)

        # Store source weights for diagnostics
        self._X_source_fit = X_source
        self._X_target_fit = X_target
        raw = self._raw_importance_weights(Xs)
        self.clip_threshold_ = float(np.quantile(raw, self.clip_quantile))
        self._raw_weights_source = raw
        self.is_fitted_ = True
        return self

    def _fit_catboost(
        self,
        X_source: NDArray[np.float64],
        X_target: NDArray[np.float64],
    ) -> None:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required for method='catboost'. "
                "Install it with: pip install catboost"
            ) from exc

        X_all = np.vstack([X_source, X_target])
        y_all = np.concatenate([
            np.zeros(len(X_source), dtype=int),
            np.ones(len(X_target), dtype=int),
        ])

        # Map categorical_cols to their positions in the feature-only matrix
        cat_indices = [
            self._feature_cols.index(c)
            for c in self.categorical_cols
            if c in self._feature_cols
        ]

        clf = CatBoostClassifier(
            iterations=self.catboost_iterations,
            depth=self.catboost_depth,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            cat_features=cat_indices if cat_indices else None,
        )
        clf.fit(X_all, y_all)
        self._model = clf
        self._feature_importances = clf.get_feature_importance()

    def _fit_rulsif(
        self,
        X_source: NDArray[np.float64],
        X_target: NDArray[np.float64],
    ) -> None:
        model = RuLSIF(alpha=self.rulsif_alpha)
        model.fit(X_source.astype(float), X_target.astype(float))
        self._model = model
        self._feature_importances = None

    def _fit_kliep(
        self,
        X_source: NDArray[np.float64],
        X_target: NDArray[np.float64],
    ) -> None:
        model = KLIEP()
        model.fit(X_source.astype(float), X_target.astype(float))
        self._model = model
        self._feature_importances = None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _raw_importance_weights(self, X_feat: NDArray) -> NDArray[np.float64]:
        """Compute weights without clipping."""
        if self.method == "catboost":
            proba = self._model.predict_proba(X_feat)  # (n, 2)
            p_target = proba[:, 1]
            p_source = proba[:, 0]
            # Avoid division by zero
            p_source = np.maximum(p_source, 1e-9)
            ratio = self._n_target / self._n_source
            return ratio * p_target / p_source
        else:
            return self._model.predict(X_feat.astype(float))

    def importance_weights(self, X: NDArray) -> NDArray[np.float64]:
        """
        Compute clipped importance weights w(x) = p_target(x) / p_source(x).

        Parameters
        ----------
        X : array-like of shape (n, d)
            Feature matrix (same columns as used in :meth:`fit`).

        Returns
        -------
        weights : array of shape (n,)
            Non-negative weights. Values are clipped at ``clip_quantile``
            to prevent single observations dominating the reweighting.

        Notes
        -----
        The weights are not normalised to sum to n. If you need normalised
        weights for a weighted estimator, divide by ``weights.mean()``.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before importance_weights().")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        Xf = X[:, self._feature_cols]
        raw = self._raw_importance_weights(Xf)

        # Apply exposure scaling if an exposure column was specified
        if self.exposure_col is not None:
            exp = X[:, self.exposure_col].astype(float)
            raw = raw * exp / (exp.mean() + 1e-12)

        # Clip
        clipped = np.minimum(raw, self.clip_threshold_)
        return np.maximum(clipped, 0.0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def shift_diagnostic(
        self,
        source_label: str = "Source",
        target_label: str = "Target",
    ) -> ShiftDiagnosticReport:
        """
        Produce a :class:`ShiftDiagnosticReport` for the fitted books.

        The report contains ESS ratio, KL divergence, per-feature driver
        analysis, and a plain-text FCA SUP 15.3 compatible summary.

        Parameters
        ----------
        source_label : str
            Human-readable label for the source book. Default "Source".
        target_label : str
            Human-readable label for the target book. Default "Target".

        Returns
        -------
        ShiftDiagnosticReport
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before shift_diagnostic().")

        return ShiftDiagnosticReport(
            weights=self._raw_weights_source,
            feature_names=self._feature_names,
            feature_importances=self._feature_importances,
            X_source=self._X_source_fit[:, self._feature_cols] if self._X_source_fit is not None else None,
            X_target=self._X_target_fit[:, self._feature_cols] if self._X_target_fit is not None else None,
            config=self.config,
            source_label=source_label,
            target_label=target_label,
        )

    # ------------------------------------------------------------------
    # sklearn interface helpers
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X_source: NDArray,
        X_target: NDArray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> NDArray[np.float64]:
        """Fit and return importance weights for X_source in one call."""
        self.fit(X_source, X_target, feature_names=feature_names)
        return self.importance_weights(X_source)

    def __repr__(self) -> str:
        return f"CovariateShiftAdaptor(method={self.method!r})"
