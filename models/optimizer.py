"""
models/optimizer.py
===================

Constrained budget allocation on top of a fitted :class:`MarketingMixModel`.

Problem
-------
Given a total budget ``B`` for a planning horizon of ``H`` weeks, choose the
dollars per channel ``b_c`` that maximise predicted revenue:

    maximise    Σ_c  Σ_{t ∈ horizon}  β_c · Hill(Adstock(x_{c,t}(b_c)))
    subject to  Σ_c b_c = B
                lo_c ≤ b_c ≤ hi_c

How a channel budget becomes a weekly spend path
------------------------------------------------
``x_{c,t}(b_c)`` re-uses the channel's *historical weekly pattern* over the
last ``H`` weeks, scaled so it sums to ``b_c``. This keeps the real flighting
(e.g. TV bursts, Q4 peaks) and therefore realistic adstock / saturation
behaviour. Weeks *before* the horizon keep their historical spend so carry-over
entering the horizon is respected. Non-media drivers (base, trend, seasonality,
controls) are unaffected by the allocation, so they are added as a constant.

Constraints
-----------
Two families of bounds are combined (the tighter one wins):

* **Share-of-budget bounds** – each channel gets between ``min_share`` and
  ``max_share`` of the total budget (default 5 %–50 %).
* **Baseline-relative bounds** – each channel stays between ``min_mult`` and
  ``max_mult`` times its *baseline* (default 0.5×–1.5×, i.e. ±50 %). The
  baseline is the historical mix scaled to the target budget, or the raw
  historical dollars if ``scale_baseline_to_budget=False``.

If the bounds are infeasible (e.g. upper bounds add up to less than the
budget), the optimizer can relax them step by step and report exactly what was
relaxed, instead of failing silently.

Solver
------
``scipy.optimize.minimize(method="SLSQP")`` on budget *shares* (well scaled),
from several starting points (Hill curves with slope > 1 are S-shaped, so the
problem is not guaranteed to be concave); the best feasible solution wins.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.mmm import MarketingMixModel, ModelNotFittedError

logger = logging.getLogger(__name__)

__all__ = ["OptimizerConstraints", "OptimizationResult", "BudgetOptimizer", "InfeasibleConstraintsError"]


class InfeasibleConstraintsError(ValueError):
    """Raised when bounds cannot all be satisfied for the requested budget."""


# =========================================================================== #
# Data classes                                                                 #
# =========================================================================== #
@dataclass
class OptimizerConstraints:
    """Bounds for the allocation.

    Attributes
    ----------
    min_share, max_share : float
        Share of the *total budget* each channel must receive (0–1).
    min_mult, max_mult : float
        Default bounds relative to each channel's baseline spend.
    channel_mult_bounds : dict[str, (float, float)]
        Per-channel overrides of (min_mult, max_mult).
    scale_baseline_to_budget : bool
        If True (default) the baseline is the historical *mix* applied to the
        new budget, so ±50 % means "±50 % vs. business-as-usual at this budget".
        If False, bounds are relative to historical dollars.
    """

    min_share: float = 0.05
    max_share: float = 0.50
    min_mult: float = 0.50
    max_mult: float = 1.50
    channel_mult_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    scale_baseline_to_budget: bool = True

    def validate(self) -> None:
        if not 0 <= self.min_share <= self.max_share <= 1:
            raise ValueError("Need 0 ≤ min_share ≤ max_share ≤ 1.")
        if not 0 <= self.min_mult <= self.max_mult:
            raise ValueError("Need 0 ≤ min_mult ≤ max_mult.")
        for ch, (lo, hi) in self.channel_mult_bounds.items():
            if not 0 <= lo <= hi:
                raise ValueError(f"Invalid multiplier bounds for '{ch}': ({lo}, {hi})")


@dataclass
class OptimizationResult:
    """Everything the dashboard needs to report an optimisation run."""

    allocation: pd.DataFrame          # one row per channel
    total_budget: float
    horizon_weeks: int
    baseline_revenue: float           # non-media revenue over the horizon (fixed)
    current_revenue: float            # predicted revenue with the baseline mix
    optimal_revenue: float            # predicted revenue with the optimal mix
    historical_revenue: float         # model-fitted revenue with actual historical spend
    success: bool
    message: str
    relaxations: List[str] = field(default_factory=list)
    n_starts: int = 0

    @property
    def lift_abs(self) -> float:
        return self.optimal_revenue - self.current_revenue

    @property
    def lift_pct(self) -> float:
        return self.lift_abs / self.current_revenue if self.current_revenue else float("nan")

    @property
    def current_media_revenue(self) -> float:
        return float(self.allocation["current_contribution"].sum())

    @property
    def optimal_media_revenue(self) -> float:
        return float(self.allocation["optimal_contribution"].sum())

    @property
    def current_roi(self) -> float:
        return self.current_media_revenue / self.total_budget if self.total_budget else float("nan")

    @property
    def optimal_roi(self) -> float:
        return self.optimal_media_revenue / self.total_budget if self.total_budget else float("nan")


# =========================================================================== #
# Optimizer                                                                    #
# =========================================================================== #
class BudgetOptimizer:
    """Revenue-maximising budget allocation for a fitted MMM.

    Parameters
    ----------
    model : MarketingMixModel
        A fitted model.
    horizon_weeks : int | None
        Planning horizon = the most recent ``horizon_weeks`` of history, whose
        weekly pattern is used for flighting. ``None`` → full history.
    """

    def __init__(self, model: MarketingMixModel, horizon_weeks: Optional[int] = 52):
        if not model.is_fitted:
            raise ModelNotFittedError("BudgetOptimizer needs a fitted MarketingMixModel.")
        self.model = model
        self.channels: List[str] = list(model.config.channel_cols)
        n = len(model.data_)
        self.horizon = int(n if horizon_weeks is None else min(max(int(horizon_weeks), 4), n))
        self._start = n - self.horizon

        # Full historical spend paths (needed for carry-over into the horizon).
        self._hist = {ch: model.data_[ch].to_numpy(dtype=float) for ch in self.channels}
        # Baseline (historical) budget per channel within the horizon.
        self.historical_spend = pd.Series(
            {ch: float(self._hist[ch][self._start:].sum()) for ch in self.channels}, name="historical_spend"
        )
        # Weekly pattern within the horizon (sums to 1); flat if the channel was dark.
        self._pattern = {}
        for ch in self.channels:
            seg = self._hist[ch][self._start:]
            total = seg.sum()
            self._pattern[ch] = seg / total if total > 0 else np.full(self.horizon, 1.0 / self.horizon)

        # Revenue from everything that is not media (fixed across scenarios).
        dec = model.decompose()
        non_media = [c for c in dec.columns if c not in (*self.channels, "fitted", "actual",
                                                         model.config.date_col)]
        self.baseline_revenue = float(dec[non_media].iloc[self._start:].to_numpy(dtype=float).sum())
        self.historical_revenue = float(dec["fitted"].iloc[self._start:].sum())
        self._coef = {ch: float(model.results_.params[ch]) for ch in self.channels}

    # ------------------------------------------------------------------ #
    # Revenue evaluation                                                   #
    # ------------------------------------------------------------------ #
    @property
    def historical_total(self) -> float:
        return float(self.historical_spend.sum())

    def channel_contribution(self, channel: str, budget: float) -> float:
        """Media revenue of ``channel`` inside the horizon for a given channel budget."""
        path = self._hist[channel].copy()
        path[self._start:] = max(budget, 0.0) * self._pattern[channel]
        contrib = self.model.channel_contribution(channel, path)
        return float(contrib[self._start:].sum())

    def media_revenue(self, budgets: np.ndarray) -> np.ndarray:
        """Per-channel contribution for a budget vector (same order as ``self.channels``)."""
        return np.array([self.channel_contribution(ch, b) for ch, b in zip(self.channels, budgets)])

    def predict_revenue(self, budgets: Dict[str, float]) -> float:
        """Total predicted revenue over the horizon for a {channel: budget} dict."""
        vec = np.array([budgets.get(ch, 0.0) for ch in self.channels], dtype=float)
        return self.baseline_revenue + float(self.media_revenue(vec).sum())

    def marginal_roi(self, budgets: np.ndarray, rel_step: float = 0.01) -> np.ndarray:
        """d(revenue)/d(spend) per channel via a forward difference."""
        out = []
        for ch, b in zip(self.channels, budgets):
            h = max(b * rel_step, 1.0)
            out.append((self.channel_contribution(ch, b + h) - self.channel_contribution(ch, b)) / h)
        return np.array(out)

    def baseline_allocation(self, total_budget: float) -> np.ndarray:
        """Historical mix applied to ``total_budget`` (business-as-usual)."""
        hist = self.historical_spend.to_numpy()
        if hist.sum() <= 0:
            return np.full(len(hist), total_budget / len(hist))
        return total_budget * hist / hist.sum()

    # ------------------------------------------------------------------ #
    # Bounds                                                               #
    # ------------------------------------------------------------------ #
    def compute_bounds(self, total_budget: float, cons: OptimizerConstraints) -> Tuple[np.ndarray, np.ndarray]:
        """Absolute dollar bounds per channel (intersection of both families)."""
        cons.validate()
        ref = (self.baseline_allocation(total_budget) if cons.scale_baseline_to_budget
               else self.historical_spend.to_numpy())
        lo, hi = [], []
        for i, ch in enumerate(self.channels):
            mlo, mhi = cons.channel_mult_bounds.get(ch, (cons.min_mult, cons.max_mult))
            lo.append(max(cons.min_share * total_budget, mlo * ref[i]))
            hi.append(min(cons.max_share * total_budget, mhi * ref[i]))
        return np.array(lo), np.array(hi)

    @staticmethod
    def _feasibility_problems(lo: np.ndarray, hi: np.ndarray, budget: float, channels: List[str]) -> List[str]:
        problems = []
        for ch, a, b in zip(channels, lo, hi):
            if a > b + 1e-6:
                problems.append(f"'{ch}': lower bound ${a:,.0f} exceeds upper bound ${b:,.0f}")
        if lo.sum() > budget * (1 + 1e-9):
            problems.append(f"lower bounds sum to ${lo.sum():,.0f} > budget ${budget:,.0f}")
        if hi.sum() < budget * (1 - 1e-9):
            problems.append(f"upper bounds sum to ${hi.sum():,.0f} < budget ${budget:,.0f}")
        return problems

    def _feasible_bounds(self, budget: float, cons: OptimizerConstraints, auto_relax: bool
                         ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return feasible bounds, relaxing step by step if allowed."""
        lo, hi = self.compute_bounds(budget, cons)
        problems = self._feasibility_problems(lo, hi, budget, self.channels)
        if not problems:
            return lo, hi, []
        if not auto_relax:
            raise InfeasibleConstraintsError("Constraints are infeasible: " + "; ".join(problems))

        relax_steps = [
            ("share-of-budget limits relaxed to 0 %–100 %",
             dict(min_share=0.0, max_share=1.0)),
            ("baseline multipliers widened to 0×–3×",
             dict(min_share=0.0, max_share=1.0, min_mult=0.0, max_mult=3.0, channel_mult_bounds={})),
            ("all bounds removed (0 to full budget)",
             dict(min_share=0.0, max_share=1.0, min_mult=0.0, max_mult=1e9, channel_mult_bounds={})),
        ]
        relaxations = [f"Original constraints infeasible ({'; '.join(problems)})."]
        for label, overrides in relax_steps:
            trial = OptimizerConstraints(**{**cons.__dict__, **overrides})
            lo, hi = self.compute_bounds(budget, trial)
            if not self._feasibility_problems(lo, hi, budget, self.channels):
                relaxations.append(f"Auto-relaxed: {label}.")
                return lo, hi, relaxations
        raise InfeasibleConstraintsError("Constraints could not be made feasible.")  # pragma: no cover

    @staticmethod
    def _project(v: np.ndarray, lo: np.ndarray, hi: np.ndarray, total: float = 1.0) -> np.ndarray:
        """Euclidean projection onto {lo ≤ x ≤ hi, Σx = total} via bisection on a shift λ."""
        a, b = (lo - v).min() - 1.0, (hi - v).max() + 1.0
        for _ in range(100):
            lam = 0.5 * (a + b)
            s = np.clip(v + lam, lo, hi).sum()
            if s > total:
                b = lam
            else:
                a = lam
        return np.clip(v + 0.5 * (a + b), lo, hi)

    # ------------------------------------------------------------------ #
    # Optimisation                                                         #
    # ------------------------------------------------------------------ #
    def optimize(
        self,
        total_budget: float,
        constraints: Optional[OptimizerConstraints] = None,
        n_random_starts: int = 4,
        auto_relax: bool = True,
        seed: int = 0,
    ) -> OptimizationResult:
        """Find the revenue-maximising allocation of ``total_budget``.

        Returns an :class:`OptimizationResult`; never raises for infeasible
        bounds when ``auto_relax`` is True (the relaxation is reported instead).
        """
        if not np.isfinite(total_budget) or total_budget <= 0:
            raise ValueError("total_budget must be a positive number.")
        cons = constraints or OptimizerConstraints()
        B = float(total_budget)
        lo, hi, relaxations = self._feasible_bounds(B, cons, auto_relax)

        # Optimise over shares s = b / B (well conditioned: all variables ~0.1–0.5).
        slo, shi = lo / B, hi / B
        current = self.baseline_allocation(B)
        current_contrib = self.media_revenue(current)
        scale = max(abs(current_contrib.sum()), 1.0)

        def objective(s: np.ndarray) -> float:
            return -float(self.media_revenue(np.clip(s, 0, None) * B).sum()) / scale

        rng = np.random.default_rng(seed)
        starts = [self._project(current / B, slo, shi),
                  self._project(np.full(len(self.channels), 1.0 / len(self.channels)), slo, shi)]
        starts += [self._project(rng.dirichlet(np.ones(len(self.channels))), slo, shi)
                   for _ in range(max(n_random_starts, 0))]

        best_x, best_val, best_res = None, np.inf, None
        eq = {"type": "eq", "fun": lambda s: np.sum(s) - 1.0, "jac": lambda s: np.ones_like(s)}
        for x0 in starts:
            try:
                with warnings.catch_warnings():
                    # SLSQP may step marginally outside the bounds before clipping – harmless.
                    warnings.filterwarnings("ignore", message="Values in x were outside bounds")
                    res = minimize(objective, x0, method="SLSQP", bounds=list(zip(slo, shi)),
                                   constraints=[eq], options={"maxiter": 500, "ftol": 1e-10})
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("SLSQP start failed: %s", exc)
                continue
            x = self._project(res.x, slo, shi)          # polish tiny bound/sum violations
            val = objective(x)
            if val < best_val:
                best_x, best_val, best_res = x, val, res

        # Never return something worse than the (projected) current mix.
        fallback = self._project(current / B, slo, shi)
        if best_x is None or objective(fallback) < best_val - 1e-12:
            best_x, best_res = fallback, best_res
        success = bool(best_res is not None and best_res.success)
        message = best_res.message if best_res is not None else "No solver run succeeded."

        optimal = best_x * B
        opt_contrib = self.media_revenue(optimal)
        alloc = pd.DataFrame({
            "channel": self.channels,
            "historical_spend": self.historical_spend.to_numpy(),
            "current_spend": current,
            "optimal_spend": optimal,
            "lower_bound": lo,
            "upper_bound": hi,
            "current_contribution": current_contrib,
            "optimal_contribution": opt_contrib,
            "current_marginal_roi": self.marginal_roi(current),
            "optimal_marginal_roi": self.marginal_roi(optimal),
        })
        alloc["spend_change"] = alloc["optimal_spend"] - alloc["current_spend"]
        alloc["spend_change_pct"] = np.where(alloc["current_spend"] > 0,
                                             alloc["spend_change"] / alloc["current_spend"], np.nan)
        alloc["current_share"] = alloc["current_spend"] / B
        alloc["optimal_share"] = alloc["optimal_spend"] / B
        alloc["current_roi"] = alloc["current_contribution"] / alloc["current_spend"].replace(0, np.nan)
        alloc["optimal_roi"] = alloc["optimal_contribution"] / alloc["optimal_spend"].replace(0, np.nan)
        tol = 1e-4 * B
        alloc["at_bound"] = np.select(
            [alloc["optimal_spend"] <= alloc["lower_bound"] + tol, alloc["optimal_spend"] >= alloc["upper_bound"] - tol],
            ["lower", "upper"], default="interior")
        alloc = alloc.set_index("channel")

        return OptimizationResult(
            allocation=alloc,
            total_budget=B,
            horizon_weeks=self.horizon,
            baseline_revenue=self.baseline_revenue,
            current_revenue=self.baseline_revenue + float(current_contrib.sum()),
            optimal_revenue=self.baseline_revenue + float(opt_contrib.sum()),
            historical_revenue=self.historical_revenue,
            success=success,
            message=str(message),
            relaxations=relaxations,
            n_starts=len(starts),
        )

    def response_curves(self, max_mult: float = 2.5, n_points: int = 41) -> pd.DataFrame:
        """Horizon contribution vs. channel budget (0 → ``max_mult`` × historical)."""
        rows = []
        for ch in self.channels:
            base = self.historical_spend[ch] if self.historical_spend[ch] > 0 else self.historical_total / len(self.channels)
            for m in np.linspace(0, max_mult, n_points):
                b = base * m
                rows.append({"channel": ch, "spend": b, "contribution": self.channel_contribution(ch, b)})
        return pd.DataFrame(rows)

    def budget_sweep(self, multipliers: np.ndarray, constraints: Optional[OptimizerConstraints] = None
                     ) -> pd.DataFrame:
        """Optimal vs. current revenue across a range of total budgets (efficient frontier)."""
        rows = []
        for m in multipliers:
            B = self.historical_total * float(m)
            r = self.optimize(B, constraints, n_random_starts=1)
            rows.append({"budget_multiplier": float(m), "total_budget": B,
                         "current_revenue": r.current_revenue, "optimal_revenue": r.optimal_revenue,
                         "current_media_revenue": r.current_media_revenue,
                         "optimal_media_revenue": r.optimal_media_revenue})
        return pd.DataFrame(rows)
