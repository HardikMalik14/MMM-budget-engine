"""
models/mmm.py
=============

Core econometric engine for the Marketing Mix Modeling (MMM) & Budget
Allocation Engine.

Pipeline for every paid-media channel
-------------------------------------

    raw weekly spend  x_t
          │
          ▼  Geometric adstock (carry-over / memory)
    A_t = x_t + α · A_{t-1}            (optionally scaled by (1 - α))
          │
          ▼  Hill saturation (diminishing returns)
    S_t = A_t^s / (A_t^s + EC50^s)     ∈ [0, 1)
          │
          ▼  OLS regression (statsmodels)
    revenue_t = β0 + Σ_c β_c · S_{c,t} + Σ_k γ_k · control_{k,t}
                + trend + Fourier seasonality + ε_t

Because every S_{c,t} lives in [0, 1), the fitted coefficient β_c has a very
practical reading: it is the *maximum weekly revenue* the channel could drive
at full saturation.

Hyper-parameters (α, EC50, slope) are not linear in the regression, so they
are either supplied by the analyst or tuned with a fast coordinate-descent grid
search that maximises adjusted R² (optionally rejecting candidates that would
give a channel a negative effect).

Public API
----------
- ``geometric_adstock``      – vectorised geometric adstock transform
- ``hill_saturation``        – Hill / sigmoid saturation transform
- ``ChannelParams``          – per-channel hyper-parameters
- ``MMMConfig``              – column mapping + model options
- ``MarketingMixModel``      – fit / predict / decompose / ROI diagnostics
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import lfilter
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

logger = logging.getLogger(__name__)

__all__ = [
    "geometric_adstock",
    "hill_saturation",
    "ChannelParams",
    "MMMConfig",
    "MarketingMixModel",
    "ModelNotFittedError",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_EC50_GRID",
    "DEFAULT_SLOPE_GRID",
]

# --------------------------------------------------------------------------- #
# Default hyper-parameter search grids                                         #
# --------------------------------------------------------------------------- #
DEFAULT_ALPHA_GRID: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_EC50_GRID: tuple = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DEFAULT_SLOPE_GRID: tuple = (0.75, 1.0, 1.5, 2.0, 2.5, 3.0)

# Small number used to protect divisions / powers from zeros.
_EPS = 1e-12


class ModelNotFittedError(RuntimeError):
    """Raised when a method that needs a fitted model is called too early."""


# =========================================================================== #
# 1. Media transformations                                                     #
# =========================================================================== #
def geometric_adstock(x: Iterable[float], alpha: float, normalize: bool = True) -> np.ndarray:
    """Apply geometric (exponential-decay) adstock to a spend series.

    The recursion is ``A_t = x_t + alpha * A_{t-1}`` with ``A_{-1} = 0``. It is
    implemented as an IIR filter via :func:`scipy.signal.lfilter`, which is
    both exact and ~100x faster than a Python loop.

    Parameters
    ----------
    x : array-like
        Raw media spend per period (must be non-negative).
    alpha : float
        Decay / retention rate in ``[0, 1)``. ``0`` means no carry-over; ``0.8``
        means 80 % of last week's (adstocked) pressure persists this week.
        The advertising half-life is ``ln(0.5) / ln(alpha)`` periods.
    normalize : bool, default True
        If True the output is multiplied by ``(1 - alpha)`` so the weights sum
        to 1. A constant spend ``s`` then converges to an adstock of exactly
        ``s``, which keeps the transformed series in "effective weekly dollars"
        and makes EC50 comparable across different decay rates.

    Returns
    -------
    np.ndarray
        Adstocked series with the same length as ``x``.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1); got {alpha!r}")
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("geometric_adstock expects a 1-D series")
    if np.isnan(arr).any():
        raise ValueError("spend series contains NaNs – clean the data first")
    out = lfilter([1.0], [1.0, -alpha], arr)
    if normalize:
        out = out * (1.0 - alpha)
    return out


def hill_saturation(x: Iterable[float], ec50: float, slope: float) -> np.ndarray:
    """Apply the Hill saturation function.

    ``S(x) = x^slope / (x^slope + ec50^slope)``

    * ``ec50``  – the (adstocked) spend level at which the channel reaches 50 %
      of its maximum effect ("half-saturation point").
    * ``slope`` – shape. ``slope <= 1`` is concave (diminishing returns from the
      first dollar); ``slope > 1`` is S-shaped (a threshold must be crossed
      before the channel "wakes up").

    The output is bounded in ``[0, 1)``.
    """
    if ec50 <= 0:
        raise ValueError(f"ec50 must be > 0; got {ec50!r}")
    if slope <= 0:
        raise ValueError(f"slope must be > 0; got {slope!r}")
    arr = np.clip(np.asarray(x, dtype=float), 0.0, None)
    # Work in ratio form (x / ec50) for numerical stability with big dollar values.
    ratio_pow = np.power(arr / ec50, slope)
    return ratio_pow / (1.0 + ratio_pow)


def adstock_half_life(alpha: float) -> float:
    """Number of periods for an ad impulse to decay to 50 % (inf if alpha=1)."""
    if alpha <= 0:
        return 0.0
    if alpha >= 1:
        return float("inf")
    return float(np.log(0.5) / np.log(alpha))


# =========================================================================== #
# 2. Configuration objects                                                     #
# =========================================================================== #
@dataclass
class ChannelParams:
    """Hyper-parameters of the adstock + Hill transformation for one channel.

    ``ec50_ratio`` is expressed *relative to the channel's mean (non-zero)
    adstocked spend* in the training data, so ``1.0`` means "the channel is at
    half-saturation at its typical spend level". The absolute dollar EC50 is
    resolved at fit time and frozen inside the model so that budget scenarios
    are evaluated on a fixed response curve.
    """

    alpha: float = 0.5
    ec50_ratio: float = 1.0
    slope: float = 1.5

    def validate(self, name: str = "channel") -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"[{name}] alpha must be in [0, 1); got {self.alpha}")
        if self.ec50_ratio <= 0:
            raise ValueError(f"[{name}] ec50_ratio must be > 0; got {self.ec50_ratio}")
        if self.slope <= 0:
            raise ValueError(f"[{name}] slope must be > 0; got {self.slope}")


@dataclass
class MMMConfig:
    """Column mapping and structural options for the regression.

    Attributes
    ----------
    target_col : str
        Dependent variable (e.g. ``total_revenue``).
    channel_cols : list[str]
        Paid-media spend columns (each gets adstock + Hill).
    control_cols : list[str]
        Non-media regressors entered linearly (macro index, holidays, price...).
    date_col : str | None
        Optional date column, used only for ordering / plotting.
    add_trend : bool
        Add a linear time trend (scaled 0→1) to absorb organic growth.
    add_seasonality : bool
        Add Fourier terms for yearly seasonality.
    n_fourier : int
        Number of sine/cosine pairs (1–4 is typical for weekly data).
    seasonal_period : float
        Length of the seasonal cycle in periods (52.18 weeks ≈ 1 year).
    normalize_adstock : bool
        See :func:`geometric_adstock`.
    """

    target_col: str
    channel_cols: List[str]
    control_cols: List[str] = field(default_factory=list)
    date_col: Optional[str] = None
    add_trend: bool = True
    add_seasonality: bool = True
    n_fourier: int = 2
    seasonal_period: float = 52.18
    normalize_adstock: bool = True

    def validate(self, df: pd.DataFrame) -> None:
        """Make sure every referenced column exists and is usable."""
        if not self.channel_cols:
            raise ValueError("At least one media channel column is required.")
        if self.target_col in self.channel_cols or self.target_col in self.control_cols:
            raise ValueError("The target column cannot also be a channel or control.")
        overlap = set(self.channel_cols) & set(self.control_cols)
        if overlap:
            raise ValueError(f"Columns used as both channel and control: {sorted(overlap)}")
        needed = [self.target_col, *self.channel_cols, *self.control_cols]
        if self.date_col:
            needed.append(self.date_col)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in data: {missing}")
        numeric = [self.target_col, *self.channel_cols, *self.control_cols]
        non_numeric = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise TypeError(f"Columns must be numeric: {non_numeric}")
        if self.n_fourier < 0:
            raise ValueError("n_fourier must be >= 0")


# =========================================================================== #
# 3. The model                                                                 #
# =========================================================================== #
class MarketingMixModel:
    """Adstock → Hill → OLS marketing-mix model with ROI diagnostics.

    Typical usage
    -------------
    >>> cfg = MMMConfig(target_col="total_revenue",
    ...                 channel_cols=["paid_search_spend", "tv_spend"],
    ...                 control_cols=["macro_economic_index", "holiday_flag"],
    ...                 date_col="date")
    >>> mmm = MarketingMixModel(cfg).fit(df, auto_tune=True)
    >>> mmm.fit_metrics()
    >>> mmm.channel_metrics()
    """

    BASE_COMPONENT = "Base (intercept + trend + seasonality)"

    def __init__(self, config: MMMConfig, channel_params: Optional[Dict[str, ChannelParams]] = None):
        self.config = config
        self.channel_params: Dict[str, ChannelParams] = {
            ch: (channel_params or {}).get(ch, ChannelParams()) for ch in config.channel_cols
        }
        for ch, p in self.channel_params.items():
            p.validate(ch)

        # Learned state (populated by fit)
        self.results_ = None                    # statsmodels RegressionResultsWrapper
        self.data_: Optional[pd.DataFrame] = None
        self.X_: Optional[pd.DataFrame] = None
        self.y_: Optional[pd.Series] = None
        self.ec50_abs_: Dict[str, float] = {}   # frozen absolute EC50 per channel (in $)
        self.control_center_: Dict[str, float] = {}  # mean used to centre continuous controls
        self.tuning_log_: List[dict] = []
        self.warnings_: List[str] = []

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self.results_ is not None

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise ModelNotFittedError("Call .fit() before using this method.")

    def _adstock(self, x: np.ndarray, channel: str, alpha: Optional[float] = None) -> np.ndarray:
        a = self.channel_params[channel].alpha if alpha is None else alpha
        return geometric_adstock(x, a, normalize=self.config.normalize_adstock)

    @staticmethod
    def _reference_level(adstocked: np.ndarray) -> float:
        """Mean of the non-zero adstocked spend – the anchor for ec50_ratio."""
        positive = adstocked[adstocked > _EPS]
        if positive.size == 0:
            return 1.0  # channel never active – any positive scale works
        return float(positive.mean())

    def _resolve_ec50(self, x: np.ndarray, channel: str, params: ChannelParams) -> float:
        adstocked = geometric_adstock(x, params.alpha, normalize=self.config.normalize_adstock)
        return params.ec50_ratio * self._reference_level(adstocked)

    def transform_channel(self, x: Iterable[float], channel: str) -> np.ndarray:
        """Adstock + Hill transform a spend series using the *fitted* EC50."""
        self._check_fitted()
        p = self.channel_params[channel]
        adstocked = self._adstock(np.asarray(x, dtype=float), channel)
        return hill_saturation(adstocked, self.ec50_abs_[channel], p.slope)

    def _structural_terms(self, n: int) -> pd.DataFrame:
        """Trend and Fourier seasonality columns for ``n`` consecutive periods."""
        cfg = self.config
        t = np.arange(n, dtype=float)
        cols: Dict[str, np.ndarray] = {}
        if cfg.add_trend:
            cols["trend"] = t / max(n - 1, 1)
        if cfg.add_seasonality and cfg.n_fourier > 0:
            for k in range(1, cfg.n_fourier + 1):
                angle = 2.0 * np.pi * k * t / cfg.seasonal_period
                cols[f"season_sin_{k}"] = np.sin(angle)
                cols[f"season_cos_{k}"] = np.cos(angle)
        return pd.DataFrame(cols)

    def _build_design(self, df: pd.DataFrame, ec50_abs: Dict[str, float],
                      params: Dict[str, ChannelParams]) -> pd.DataFrame:
        """Full regression design matrix (with constant) for given hyper-params."""
        cfg = self.config
        parts = {}
        for ch in cfg.channel_cols:
            p = params[ch]
            adstocked = geometric_adstock(df[ch].to_numpy(dtype=float), p.alpha,
                                          normalize=cfg.normalize_adstock)
            parts[ch] = hill_saturation(adstocked, ec50_abs[ch], p.slope)
        X = pd.DataFrame(parts, index=df.index)
        for c in cfg.control_cols:
            # Continuous controls are centred on their training mean so that their
            # contribution is the effect of *deviating* from normal conditions and
            # the intercept keeps the baseline level. 0/1 flags stay uncentred.
            X[c] = df[c].to_numpy(dtype=float) - self.control_center_.get(c, 0.0)
        struct = self._structural_terms(len(df))
        struct.index = df.index
        X = pd.concat([X, struct], axis=1)
        X = sm.add_constant(X, has_constant="add")
        return X

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, sort by date and clean the modelling frame."""
        cfg = self.config
        cfg.validate(df)
        data = df.copy()
        if cfg.date_col:
            data[cfg.date_col] = pd.to_datetime(data[cfg.date_col], errors="coerce")
            if data[cfg.date_col].isna().any():
                raise ValueError(f"Some values in '{cfg.date_col}' could not be parsed as dates.")
            data = data.sort_values(cfg.date_col)
        data = data.reset_index(drop=True)

        # Spend: missing → 0, negative → 0 (with a warning)
        for ch in cfg.channel_cols:
            n_nan = int(data[ch].isna().sum())
            if n_nan:
                self.warnings_.append(f"'{ch}': {n_nan} missing spend values filled with 0.")
                data[ch] = data[ch].fillna(0.0)
            n_neg = int((data[ch] < 0).sum())
            if n_neg:
                self.warnings_.append(f"'{ch}': {n_neg} negative spend values clipped to 0.")
                data[ch] = data[ch].clip(lower=0.0)

        # Controls: forward/back fill short gaps; Target: rows with NaN dropped
        for c in cfg.control_cols:
            if data[c].isna().any():
                self.warnings_.append(f"'{c}': missing values forward/back-filled.")
                data[c] = data[c].ffill().bfill()
        n_before = len(data)
        data = data.dropna(subset=[cfg.target_col]).reset_index(drop=True)
        if len(data) < n_before:
            self.warnings_.append(f"Dropped {n_before - len(data)} rows with a missing target.")

        n_params = 1 + len(cfg.channel_cols) + len(cfg.control_cols) + int(cfg.add_trend) \
            + (2 * cfg.n_fourier if cfg.add_seasonality else 0)
        if len(data) <= n_params + 5:
            raise ValueError(
                f"Not enough observations ({len(data)}) for {n_params} parameters. "
                "Use fewer controls / Fourier terms or more data."
            )
        if len(data) < 52:
            self.warnings_.append(
                f"Only {len(data)} periods of data – MMM estimates are usually unstable below ~1 year."
            )
        for ch in cfg.channel_cols:
            if float(data[ch].sum()) <= 0:
                raise ValueError(f"Channel '{ch}' has zero total spend – remove it from the model.")
        return data

    # ------------------------------------------------------------------ #
    # Hyper-parameter tuning                                               #
    # ------------------------------------------------------------------ #
    def tune_hyperparameters(
        self,
        df: pd.DataFrame,
        alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
        ec50_grid: Sequence[float] = DEFAULT_EC50_GRID,
        slope_grid: Sequence[float] = DEFAULT_SLOPE_GRID,
        n_passes: int = 2,
        require_positive: bool = True,
    ) -> Dict[str, ChannelParams]:
        """Coordinate-descent grid search over (alpha, ec50_ratio, slope).

        Each pass cycles through the channels; for one channel at a time it
        evaluates every grid combination while holding the other channels at
        their current best, and keeps the combination with the highest adjusted
        R² (all candidates have the same parameter count, so this is equivalent
        to minimising SSE / AIC). With ``require_positive`` a candidate that
        makes the channel's coefficient negative is rejected – negative media
        effects are rarely plausible and break budget optimisation.

        Uses ``numpy.linalg.lstsq`` for speed (thousands of fits per second);
        the final model is re-estimated with statsmodels in :meth:`fit`.
        """
        data = self._prepare(df)
        cfg = self.config
        y = data[cfg.target_col].to_numpy(dtype=float)
        n = len(y)
        sst = float(((y - y.mean()) ** 2).sum()) or 1.0

        # Pre-compute everything that does not depend on the channel being tuned.
        fixed_cols = [data[c].to_numpy(dtype=float) for c in cfg.control_cols]
        struct = self._structural_terms(n)
        fixed_cols += [struct[c].to_numpy() for c in struct.columns]
        fixed_block = np.column_stack([np.ones(n), *fixed_cols]) if fixed_cols else np.ones((n, 1))
        k_total = fixed_block.shape[1] + len(cfg.channel_cols)

        spend = {ch: data[ch].to_numpy(dtype=float) for ch in cfg.channel_cols}
        best = {ch: ChannelParams(**asdict(self.channel_params[ch])) for ch in cfg.channel_cols}

        def feature(ch: str, p: ChannelParams) -> np.ndarray:
            ad = geometric_adstock(spend[ch], p.alpha, normalize=cfg.normalize_adstock)
            return hill_saturation(ad, p.ec50_ratio * self._reference_level(ad), p.slope)

        current = {ch: feature(ch, best[ch]) for ch in cfg.channel_cols}
        self.tuning_log_ = []

        for pass_idx in range(max(1, n_passes)):
            for j, ch in enumerate(cfg.channel_cols):
                others = [current[o] for o in cfg.channel_cols if o != ch]
                base_block = np.column_stack([fixed_block, *others]) if others else fixed_block
                best_score, best_params, best_any = -np.inf, None, (-np.inf, None)

                # Adstock depends only on alpha → compute once per alpha.
                for a in alpha_grid:
                    ad = geometric_adstock(spend[ch], a, normalize=cfg.normalize_adstock)
                    ref = self._reference_level(ad)
                    for r in ec50_grid:
                        for s in slope_grid:
                            f = hill_saturation(ad, r * ref, s)
                            X = np.column_stack([base_block, f])
                            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                            resid = y - X @ coef
                            r2 = 1.0 - float(resid @ resid) / sst
                            adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k_total, 1)
                            if adj > best_any[0]:
                                best_any = (adj, ChannelParams(a, r, s))
                            if require_positive and coef[-1] <= 0:
                                continue
                            if adj > best_score:
                                best_score, best_params = adj, ChannelParams(a, r, s)

                if best_params is None:  # every candidate gave a negative coefficient
                    best_score, best_params = best_any
                    msg = (f"'{ch}': no hyper-parameters give a positive effect; "
                           "keeping best fit regardless (check data / collinearity).")
                    if msg not in self.warnings_:
                        self.warnings_.append(msg)
                best[ch] = best_params
                current[ch] = feature(ch, best_params)
                self.tuning_log_.append({"pass": pass_idx + 1, "channel": ch,
                                         **asdict(best_params), "adj_r2": best_score})
                logger.debug("pass %d %s -> %s (adjR2=%.4f)", pass_idx + 1, ch, best_params, best_score)

        self.channel_params = best
        return best

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, auto_tune: bool = False, **tune_kwargs) -> "MarketingMixModel":
        """Estimate the model.

        Parameters
        ----------
        df : DataFrame
            Weekly (or other regular-period) data containing all configured columns.
        auto_tune : bool
            If True, run :meth:`tune_hyperparameters` first.
        **tune_kwargs
            Forwarded to :meth:`tune_hyperparameters` (grids, n_passes, ...).
        """
        self.warnings_ = []
        if auto_tune:
            self.tune_hyperparameters(df, **tune_kwargs)
        data = self._prepare(df)
        cfg = self.config

        # Centre continuous controls (binary flags are left as 0/1).
        self.control_center_ = {}
        for c in cfg.control_cols:
            vals = set(np.unique(data[c].dropna().to_numpy()))
            self.control_center_[c] = 0.0 if vals <= {0.0, 1.0} else float(data[c].mean())

        # Freeze absolute EC50 (in adstocked $) from the training data.
        self.ec50_abs_ = {
            ch: self._resolve_ec50(data[ch].to_numpy(dtype=float), ch, self.channel_params[ch])
            for ch in cfg.channel_cols
        }
        X = self._build_design(data, self.ec50_abs_, self.channel_params)
        y = data[cfg.target_col].astype(float)

        # Drop columns with zero variance (e.g. a holiday flag that is never 1).
        constant_cols = [c for c in X.columns if c != "const" and float(X[c].std()) < 1e-12]
        if constant_cols:
            media_const = [c for c in constant_cols if c in cfg.channel_cols]
            if media_const:
                raise ValueError(f"Transformed media columns are constant: {media_const}. "
                                 "Adjust EC50 / slope.")
            self.warnings_.append(f"Dropped constant regressors: {constant_cols}")
            X = X.drop(columns=constant_cols)

        try:
            self.results_ = sm.OLS(y, X).fit()
        except Exception as exc:  # pragma: no cover - statsmodels rarely fails here
            raise RuntimeError(f"OLS estimation failed: {exc}") from exc

        self.data_, self.X_, self.y_ = data, X, y
        self._post_fit_checks()
        return self

    def _post_fit_checks(self) -> None:
        """Collect human-readable diagnostics warnings."""
        res = self.results_
        for ch in self.config.channel_cols:
            coef, p = float(res.params[ch]), float(res.pvalues[ch])
            if coef < 0:
                self.warnings_.append(
                    f"'{ch}' has a NEGATIVE coefficient ({coef:,.0f}). The optimizer will "
                    "push it to its minimum bound – investigate collinearity or data issues."
                )
            elif p > 0.10:
                self.warnings_.append(f"'{ch}' is not statistically significant (p = {p:.3f}).")
        dw = float(durbin_watson(res.resid))
        if dw < 1.5 or dw > 2.5:
            self.warnings_.append(
                f"Durbin-Watson = {dw:.2f}: residuals are autocorrelated – consider more "
                "seasonality terms or additional controls."
            )

    # ------------------------------------------------------------------ #
    # Prediction & decomposition                                          #
    # ------------------------------------------------------------------ #
    def predict(self, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Predict the target for ``df`` (defaults to the training data).

        Note: adstock is path-dependent, so ``df`` should be a contiguous block
        of periods ordered in time. Trend / seasonality are re-generated from
        period 0 of ``df``; to score a future window pass history + future and
        slice the result.
        """
        self._check_fitted()
        if df is None:
            return np.asarray(self.results_.fittedvalues, dtype=float)
        data = df.copy().reset_index(drop=True)
        X = self._build_design(data, self.ec50_abs_, self.channel_params)
        X = X[self.X_.columns]
        return np.asarray(X.to_numpy(dtype=float) @ self.results_.params.to_numpy(), dtype=float)

    def decompose(self) -> pd.DataFrame:
        """Additive decomposition of the fitted values over time.

        Returns a DataFrame with one column per component: the base (intercept,
        trend and seasonality), each control variable and each channel. Rows
        sum exactly to the fitted values.
        """
        self._check_fitted()
        params = self.results_.params
        X = self.X_
        cfg = self.config
        out = pd.DataFrame(index=X.index)
        base_cols = [c for c in X.columns if c == "const" or c == "trend" or c.startswith("season_")]
        out[self.BASE_COMPONENT] = (X[base_cols] * params[base_cols]).sum(axis=1)
        for c in cfg.control_cols:
            if c in X.columns:
                out[c] = X[c] * params[c]
        for ch in cfg.channel_cols:
            out[ch] = X[ch] * params[ch]
        if cfg.date_col:
            out.insert(0, cfg.date_col, self.data_[cfg.date_col].to_numpy())
        out["fitted"] = self.results_.fittedvalues.to_numpy()
        out["actual"] = self.y_.to_numpy()
        return out

    def channel_contribution(self, channel: str, spend: Iterable[float]) -> np.ndarray:
        """Weekly revenue contribution of ``channel`` for an arbitrary spend path."""
        self._check_fitted()
        return float(self.results_.params[channel]) * self.transform_channel(spend, channel)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #
    def fit_metrics(self) -> Dict[str, float]:
        """Goodness-of-fit statistics of the in-sample fit."""
        self._check_fitted()
        res = self.results_
        y = self.y_.to_numpy()
        yhat = res.fittedvalues.to_numpy()
        nonzero = np.abs(y) > _EPS
        return {
            "r2": float(r2_score(y, yhat)),
            "adj_r2": float(res.rsquared_adj),
            "mape": float(mean_absolute_percentage_error(y[nonzero], yhat[nonzero])) if nonzero.any() else np.nan,
            "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "durbin_watson": float(durbin_watson(res.resid)),
            "f_pvalue": float(res.f_pvalue) if res.f_pvalue is not None else np.nan,
            "aic": float(res.aic),
            "bic": float(res.bic),
            "n_obs": int(res.nobs),
            "n_params": int(len(res.params)),
        }

    def coefficient_table(self) -> pd.DataFrame:
        """Coefficients, standard errors, t-stats, p-values, 95 % CI and VIF."""
        self._check_fitted()
        res = self.results_
        ci = res.conf_int(alpha=0.05)
        table = pd.DataFrame({
            "coefficient": res.params,
            "std_error": res.bse,
            "t_stat": res.tvalues,
            "p_value": res.pvalues,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
        })
        # Variance inflation factors (multicollinearity) – skip the constant.
        Xv = self.X_.to_numpy(dtype=float)
        vifs = {}
        for i, col in enumerate(self.X_.columns):
            if col == "const":
                vifs[col] = np.nan
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    vifs[col] = float(variance_inflation_factor(Xv, i))
                except Exception:
                    vifs[col] = np.nan
        table["vif"] = pd.Series(vifs)

        def kind(c: str) -> str:
            if c in self.config.channel_cols:
                return "media"
            if c in self.config.control_cols:
                return "control"
            return "structural"

        table.insert(0, "type", [kind(c) for c in table.index])
        table["significant_5pct"] = table["p_value"] < 0.05
        table.index.name = "variable"
        return table

    def channel_metrics(self) -> pd.DataFrame:
        """Per-channel spend, contribution, contribution share, ROI and marginal ROI.

        * ``contribution``             – Σ_t β_c · S_{c,t} (revenue attributed to the channel)
        * ``contribution_pct_revenue`` – share of total fitted revenue
        * ``contribution_pct_media``   – share of all media-driven revenue
        * ``roi``                      – contribution / spend (revenue per $1)
        * ``marginal_roi``             – extra revenue from the *next* $1 spent,
                                         estimated by scaling spend by +1 %
        """
        self._check_fitted()
        cfg = self.config
        total_fitted = float(self.results_.fittedvalues.sum())
        rows = []
        for ch in cfg.channel_cols:
            spend = self.data_[ch].to_numpy(dtype=float)
            total_spend = float(spend.sum())
            contrib = float(self.channel_contribution(ch, spend).sum())
            bumped = float(self.channel_contribution(ch, spend * 1.01).sum())
            p = self.channel_params[ch]
            rows.append({
                "channel": ch,
                "spend": total_spend,
                "contribution": contrib,
                "roi": contrib / total_spend if total_spend > 0 else np.nan,
                "marginal_roi": (bumped - contrib) / (0.01 * total_spend) if total_spend > 0 else np.nan,
                "coefficient": float(self.results_.params[ch]),
                "p_value": float(self.results_.pvalues[ch]),
                "alpha": p.alpha,
                "half_life_weeks": adstock_half_life(p.alpha),
                "ec50_ratio": p.ec50_ratio,
                "ec50_abs": self.ec50_abs_[ch],
                "slope": p.slope,
            })
        out = pd.DataFrame(rows).set_index("channel")
        media_total = float(out["contribution"].sum())
        out.insert(2, "contribution_pct_revenue", out["contribution"] / total_fitted if total_fitted else np.nan)
        out.insert(3, "contribution_pct_media", out["contribution"] / media_total if media_total else np.nan)
        out.insert(1, "spend_share", out["spend"] / out["spend"].sum())
        return out

    def response_curve(self, channel: str, multipliers: Optional[Sequence[float]] = None) -> pd.DataFrame:
        """Total contribution of ``channel`` when its historical spend path is
        scaled by each multiplier (0 → 3x by default). Carry-over and flighting
        are preserved because the whole weekly pattern is scaled."""
        self._check_fitted()
        if multipliers is None:
            multipliers = np.linspace(0.0, 3.0, 61)
        spend = self.data_[channel].to_numpy(dtype=float)
        rows = []
        for m in multipliers:
            contrib = float(self.channel_contribution(channel, spend * m).sum())
            rows.append({"multiplier": float(m), "spend": float(spend.sum() * m), "contribution": contrib})
        return pd.DataFrame(rows)

    def holdout_validation(self, test_fraction: float = 0.2) -> Dict[str, float]:
        """Out-of-time check: re-estimate the linear coefficients on the first
        ``1 - test_fraction`` of periods (keeping the fitted hyper-parameters)
        and score the most recent periods.

        Media transforms are computed on the full series so adstock carry-over
        into the test window is respected. EC50 scales come from the full fit
        (a small, documented look-ahead that only affects the curve's x-scale).
        """
        self._check_fitted()
        if not 0.05 <= test_fraction <= 0.5:
            raise ValueError("test_fraction must be between 0.05 and 0.5")
        X, y = self.X_, self.y_
        n_test = max(int(round(len(y) * test_fraction)), 4)
        n_train = len(y) - n_test
        if n_train <= X.shape[1] + 5:
            raise ValueError("Not enough training rows for a hold-out split.")
        Xtr, ytr, Xte, yte = X.iloc[:n_train], y.iloc[:n_train], X.iloc[n_train:], y.iloc[n_train:]
        # Columns that are constant within the training window cannot be estimated.
        keep = [c for c in X.columns if c == "const" or float(Xtr[c].std()) > 1e-12]
        res = sm.OLS(ytr, Xtr[keep]).fit()
        pred = res.predict(Xte[keep]).to_numpy()
        ytrue = yte.to_numpy()
        nz = np.abs(ytrue) > _EPS
        return {
            "n_train": int(n_train),
            "n_test": int(n_test),
            "train_r2": float(res.rsquared),
            "test_r2": float(r2_score(ytrue, pred)),
            "test_mape": float(mean_absolute_percentage_error(ytrue[nz], pred[nz])) if nz.any() else np.nan,
            "test_rmse": float(np.sqrt(mean_squared_error(ytrue, pred))),
        }

    def summary_text(self) -> str:
        """Full statsmodels regression summary as plain text."""
        self._check_fitted()
        return str(self.results_.summary())

    def params_table(self) -> pd.DataFrame:
        """Current hyper-parameters as a tidy table."""
        rows = []
        for ch, p in self.channel_params.items():
            rows.append({"channel": ch, "alpha": p.alpha, "half_life_weeks": adstock_half_life(p.alpha),
                         "ec50_ratio": p.ec50_ratio, "ec50_abs": self.ec50_abs_.get(ch, np.nan),
                         "slope": p.slope})
        return pd.DataFrame(rows).set_index("channel")
