"""
data/generator.py
=================

Synthetic weekly marketing dataset with a *known* data-generating process.

Why synthetic data?  Real MMM data is confidential, and a known ground truth is
the only way to check that the engine recovers the right carry-over, saturation
and ROI. Every number used to create the revenue series is saved next to the
CSV (``synthetic_ground_truth.json``) and surfaced in the dashboard.

Output columns
--------------
date                   Week start (Monday)
week                   1..N week index
paid_search_spend      Always-on, follows demand (higher in Q4)
paid_social_spend      Always-on with growth + periodic campaign pulses
influencer_spend       Sporadic 2–4 week campaigns, near zero otherwise
tv_spend               Flighted bursts (on/off), heavy in Q4
macro_economic_index   Mean-reverting consumer-confidence style index (~100)
holiday_flag           1 if the week contains a major US retail holiday
total_revenue          Target variable

Revenue process
---------------
revenue_t = base + trend·t + seasonality_t + holiday lift + macro effect
            + Σ_c β_c · Hill(Adstock(spend_c; α_c); EC50_c, slope_c)
            + ε_t,     ε_t ~ N(0, σ²)  with mild AR(1) persistence

Run ``python data/generator.py`` (or ``python -m data.generator``) to write
``data/synthetic_mmm_data.csv`` and the ground-truth JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Allow ``python data/generator.py`` as well as ``python -m data.generator``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.mmm import geometric_adstock, hill_saturation  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = DATA_DIR / "synthetic_mmm_data.csv"
DEFAULT_TRUTH = DATA_DIR / "synthetic_ground_truth.json"

CHANNELS = ["paid_search_spend", "paid_social_spend", "influencer_spend", "tv_spend"]

# --------------------------------------------------------------------------- #
# Ground-truth media parameters                                                #
#   alpha       adstock retention (TV remembers longest, search shortest)      #
#   ec50_ratio  half-saturation point relative to mean non-zero adstock        #
#   slope       Hill shape (TV is S-shaped, influencer is concave)             #
#   beta        revenue per week at full saturation                            #
# --------------------------------------------------------------------------- #
TRUE_MEDIA_PARAMS: Dict[str, Dict[str, float]] = {
    "paid_search_spend": {"alpha": 0.20, "ec50_ratio": 1.00, "slope": 1.0, "beta": 130_000.0},
    "paid_social_spend": {"alpha": 0.40, "ec50_ratio": 1.50, "slope": 1.5, "beta": 95_000.0},
    "influencer_spend":  {"alpha": 0.50, "ec50_ratio": 0.75, "slope": 1.0, "beta": 40_000.0},
    "tv_spend":          {"alpha": 0.70, "ec50_ratio": 1.00, "slope": 2.0, "beta": 150_000.0},
}

BASE_REVENUE = 420_000.0         # weekly organic revenue at t = 0
TREND_PER_WEEK = 450.0           # organic growth per week
SEASONAL_AMPLITUDE = 45_000.0    # main yearly cycle
HOLIDAY_LIFT = 60_000.0          # extra revenue in holiday weeks
MACRO_COEF = 4_000.0             # revenue per index point above 100
NOISE_PCT = 0.025                # residual noise as % of mean revenue


# =========================================================================== #
# Calendar helpers                                                             #
# =========================================================================== #
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th ``weekday`` (Mon=0) of a month; n = -1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def us_retail_holidays(year: int) -> list[date]:
    """Major US retail moments: New Year, Memorial Day, July 4th, Labor Day,
    Thanksgiving, Black Friday, Cyber Monday and Christmas."""
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    return [
        date(year, 1, 1),
        _nth_weekday(year, 5, 0, -1),          # Memorial Day
        date(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),           # Labor Day
        thanksgiving,
        thanksgiving + timedelta(days=1),      # Black Friday
        thanksgiving + timedelta(days=4),      # Cyber Monday
        date(year, 12, 25),
    ]


def holiday_flags(week_starts: pd.DatetimeIndex) -> np.ndarray:
    """1 if a holiday falls inside [week_start, week_start + 6 days]."""
    years = range(week_starts.min().year - 1, week_starts.max().year + 2)
    holidays = {h for y in years for h in us_retail_holidays(y)}
    flags = []
    for ws in week_starts:
        d0 = ws.date()
        flags.append(int(any(d0 + timedelta(days=k) in holidays for k in range(7))))
    return np.array(flags, dtype=int)


# =========================================================================== #
# Spend simulators                                                             #
# =========================================================================== #
def _seasonal_index(week_starts: pd.DatetimeIndex) -> np.ndarray:
    """Demand seasonality in [-1, 1]: trough in late winter, peak in Nov/Dec."""
    doy = week_starts.dayofyear.to_numpy(dtype=float)
    # Phase chosen so the peak lands around week 48 (late November).
    return np.cos(2 * np.pi * (doy - 335) / 365.25)


def _simulate_search(rng, season, n) -> np.ndarray:
    level = 38_000 * (1 + 0.12 * season) * (1 + 0.0010 * np.arange(n))
    spend = level * rng.lognormal(0.0, 0.25, n)
    # Budget-cap / throttling weeks (~8 % of weeks at 15-35 % of normal spend).
    # Real accounts have these too, and they are what lets an MMM separate a
    # channel's baseline effect from the intercept (identifiability).
    throttled = rng.random(n) < 0.08
    spend[throttled] *= rng.uniform(0.15, 0.35, throttled.sum())
    return spend


def _simulate_social(rng, season, n) -> np.ndarray:
    level = 26_000 * (1 + 0.0015 * np.arange(n)) * (1 + 0.05 * season)
    spend = level * rng.lognormal(0.0, 0.25, n)
    # Quarterly 3-week campaign pulses (+60 %)
    for start in range(6, n, 13):
        spend[start:start + 3] *= 1.6
    # Occasional creative-refresh pauses (1-2 weeks at ~10 % of normal spend)
    for start in rng.choice(np.arange(n - 2), size=max(n // 26, 1), replace=False):
        spend[start:start + int(rng.integers(1, 3))] *= 0.10
    return spend


def _simulate_influencer(rng, n) -> np.ndarray:
    spend = rng.uniform(0, 1_500, n)          # small always-on seeding
    t = 0
    while t < n:
        t += int(rng.integers(4, 9))          # gap between campaigns
        length = int(rng.integers(2, 5))      # 2–4 week campaigns
        spend[t:t + length] += rng.uniform(20_000, 55_000)
        t += length
    return spend


def _simulate_tv(rng, season, week_starts, n) -> np.ndarray:
    spend = np.zeros(n)
    t = int(rng.integers(0, 4))
    while t < n:
        length = int(rng.integers(3, 7))      # 3–6 week flights
        spend[t:t + length] = rng.uniform(70_000, 130_000, size=min(length, n - t))
        t += length + int(rng.integers(3, 8))  # dark period between flights
    # Guaranteed heavy Q4 flight (weeks 44–51 of each year)
    woy = week_starts.isocalendar().week.to_numpy()
    q4 = (woy >= 44) & (woy <= 51)
    spend[q4] = np.maximum(spend[q4], rng.uniform(110_000, 160_000, q4.sum()))
    return spend * (1 + 0.1 * season)


def _simulate_macro(rng, n) -> np.ndarray:
    """Mean-reverting (AR(1)) index around a gently rising level."""
    x = np.empty(n)
    x[0] = 100.0
    for t in range(1, n):
        target = 100.0 + 0.02 * t
        x[t] = x[t - 1] + 0.15 * (target - x[t - 1]) + rng.normal(0, 0.9)
    return x


# =========================================================================== #
# Main generator                                                               #
# =========================================================================== #
def generate_synthetic_data(
    n_weeks: int = 156,
    start_date: str = "2023-01-02",
    seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """Create the synthetic dataset and its ground truth.

    Parameters
    ----------
    n_weeks : int
        Number of weekly observations (default 156 = 3 years).
    start_date : str
        First week start (should be a Monday).
    seed : int
        Random seed for full reproducibility.

    Returns
    -------
    (DataFrame, dict)
        The dataset and a JSON-serialisable dict with the true parameters,
        true contributions and true ROI per channel.
    """
    if n_weeks < 30:
        raise ValueError("n_weeks should be at least 30 for a meaningful MMM dataset.")
    rng = np.random.default_rng(seed)
    week_starts = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks, dtype=float)
    season = _seasonal_index(week_starts)

    spend = {
        "paid_search_spend": _simulate_search(rng, season, n_weeks),
        "paid_social_spend": _simulate_social(rng, season, n_weeks),
        "influencer_spend": _simulate_influencer(rng, n_weeks),
        "tv_spend": _simulate_tv(rng, season, week_starts, n_weeks),
    }
    macro = _simulate_macro(rng, n_weeks)
    holidays = holiday_flags(week_starts)

    # ---- Organic (non-media) revenue --------------------------------------
    base = BASE_REVENUE + TREND_PER_WEEK * t
    seasonality = SEASONAL_AMPLITUDE * season + 0.25 * SEASONAL_AMPLITUDE * np.sin(4 * np.pi * t / 52.18)
    holiday_effect = HOLIDAY_LIFT * holidays
    macro_effect = MACRO_COEF * (macro - 100.0)

    # ---- Media revenue (adstock -> Hill -> beta) ---------------------------
    media_contrib: Dict[str, np.ndarray] = {}
    resolved_ec50: Dict[str, float] = {}
    for ch, p in TRUE_MEDIA_PARAMS.items():
        ad = geometric_adstock(spend[ch], p["alpha"], normalize=True)
        ref = float(ad[ad > 1e-12].mean())
        ec50 = p["ec50_ratio"] * ref
        resolved_ec50[ch] = ec50
        media_contrib[ch] = p["beta"] * hill_saturation(ad, ec50, p["slope"])

    # ---- Noise with mild autocorrelation -----------------------------------
    clean = base + seasonality + holiday_effect + macro_effect + sum(media_contrib.values())
    sigma = NOISE_PCT * clean.mean()
    eps = np.empty(n_weeks)
    eps[0] = rng.normal(0, sigma)
    for i in range(1, n_weeks):
        eps[i] = 0.3 * eps[i - 1] + rng.normal(0, sigma * np.sqrt(1 - 0.3 ** 2))
    revenue = clean + eps

    df = pd.DataFrame({
        "date": week_starts,
        "week": np.arange(1, n_weeks + 1),
        **{ch: np.round(v, 2) for ch, v in spend.items()},
        "macro_economic_index": np.round(macro, 3),
        "holiday_flag": holidays,
        "total_revenue": np.round(revenue, 2),
    })

    truth = {
        "seed": seed,
        "n_weeks": n_weeks,
        "description": "Ground-truth data-generating process for synthetic_mmm_data.csv",
        "organic": {
            "base_revenue": BASE_REVENUE,
            "trend_per_week": TREND_PER_WEEK,
            "seasonal_amplitude": SEASONAL_AMPLITUDE,
            "holiday_lift": HOLIDAY_LIFT,
            "macro_coef_per_point": MACRO_COEF,
            "noise_sigma": float(sigma),
        },
        "channels": {
            ch: {
                **TRUE_MEDIA_PARAMS[ch],
                "ec50_abs": resolved_ec50[ch],
                "total_spend": float(spend[ch].sum()),
                "true_contribution": float(media_contrib[ch].sum()),
                "true_roi": float(media_contrib[ch].sum() / spend[ch].sum()),
                "true_contribution_pct_revenue": float(media_contrib[ch].sum() / revenue.sum()),
            }
            for ch in CHANNELS
        },
    }
    return df, truth


def save_synthetic_data(
    csv_path: Path = DEFAULT_CSV,
    truth_path: Path = DEFAULT_TRUTH,
    **kwargs,
) -> pd.DataFrame:
    """Generate and write the CSV + ground-truth JSON. Returns the DataFrame."""
    df, truth = generate_synthetic_data(**kwargs)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    Path(truth_path).write_text(json.dumps(truth, indent=2))
    return df


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MMM weekly data.")
    parser.add_argument("--weeks", type=int, default=156, help="number of weeks (default 156)")
    parser.add_argument("--start", default="2023-01-02", help="first Monday (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", default=str(DEFAULT_CSV), help="output CSV path")
    parser.add_argument("--truth", default=str(DEFAULT_TRUTH), help="ground-truth JSON path")
    args = parser.parse_args()

    df = save_synthetic_data(Path(args.out), Path(args.truth),
                             n_weeks=args.weeks, start_date=args.start, seed=args.seed)
    print(f"Wrote {len(df)} weeks to {args.out}")
    print(f"Ground truth written to {args.truth}\n")
    print(df.head().to_string(index=False))
    truth = json.loads(Path(args.truth).read_text())
    print("\nTrue channel ROI:")
    for ch, info in truth["channels"].items():
        print(f"  {ch:<20} ROI = {info['true_roi']:.2f}   "
              f"share of revenue = {info['true_contribution_pct_revenue']:.1%}")


if __name__ == "__main__":
    _cli()
