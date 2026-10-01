"""
Unit tests for the MMM engine, optimizer and data layer.

Run:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.generator import CHANNELS, generate_synthetic_data
from data.loader import coerce_numeric_columns, detect_schema, prepare_dataframe
from models.mmm import (
    ChannelParams,
    MarketingMixModel,
    MMMConfig,
    ModelNotFittedError,
    adstock_half_life,
    geometric_adstock,
    hill_saturation,
)
from models.optimizer import BudgetOptimizer, InfeasibleConstraintsError, OptimizerConstraints

CONTROLS = ["macro_economic_index", "holiday_flag"]


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic():
    df, truth = generate_synthetic_data(n_weeks=156, seed=42)
    return df, truth


@pytest.fixture(scope="module")
def fitted(synthetic):
    df, _ = synthetic
    cfg = MMMConfig(target_col="total_revenue", channel_cols=CHANNELS, control_cols=CONTROLS, date_col="date")
    return MarketingMixModel(cfg).fit(df, auto_tune=True)


# --------------------------------------------------------------------------- #
# Transformations                                                              #
# --------------------------------------------------------------------------- #
def test_adstock_matches_recursion():
    x = np.array([100.0, 0, 0, 50, 0])
    alpha = 0.6
    expected = np.zeros_like(x)
    for t in range(len(x)):
        expected[t] = x[t] + (alpha * expected[t - 1] if t else 0)
    np.testing.assert_allclose(geometric_adstock(x, alpha, normalize=False), expected)
    np.testing.assert_allclose(geometric_adstock(x, alpha, normalize=True), expected * (1 - alpha))


def test_adstock_steady_state_normalized():
    out = geometric_adstock(np.full(400, 1000.0), 0.8, normalize=True)
    assert out[-1] == pytest.approx(1000.0, rel=1e-6)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_adstock_rejects_bad_alpha(bad):
    with pytest.raises(ValueError):
        geometric_adstock([1, 2, 3], bad)


def test_hill_properties():
    x = np.linspace(0, 10_000, 101)
    y = hill_saturation(x, ec50=2_000, slope=1.5)
    assert y[0] == 0
    assert np.all(np.diff(y) >= 0)                       # monotone
    assert np.all((y >= 0) & (y < 1))                    # bounded
    assert hill_saturation([2_000], 2_000, 3.0)[0] == pytest.approx(0.5)  # half-saturation


def test_half_life():
    assert adstock_half_life(0.5) == pytest.approx(1.0)
    assert adstock_half_life(0.0) == 0.0


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def test_model_fit_quality(fitted):
    m = fitted.fit_metrics()
    assert m["r2"] > 0.9
    assert m["mape"] < 0.05


def test_decomposition_sums_to_fitted(fitted):
    dec = fitted.decompose()
    parts = dec.drop(columns=["date", "fitted", "actual"]).sum(axis=1)
    np.testing.assert_allclose(parts.to_numpy(), dec["fitted"].to_numpy(), rtol=1e-8)


def test_channel_metrics_and_recovery(fitted, synthetic):
    _, truth = synthetic
    cm = fitted.channel_metrics()
    assert set(cm.index) == set(CHANNELS)
    assert (cm["contribution"] > 0).all()
    assert cm["contribution_pct_media"].sum() == pytest.approx(1.0)
    # Recovered ROI should be in the right ballpark of the truth (within 2x).
    for ch in CHANNELS:
        true_roi = truth["channels"][ch]["true_roi"]
        assert 0.5 * true_roi < cm.loc[ch, "roi"] < 2.0 * true_roi, ch


def test_predict_reproduces_fitted(synthetic, fitted):
    df, _ = synthetic
    np.testing.assert_allclose(fitted.predict(df), fitted.predict(), rtol=1e-8)


def test_holdout_validation(fitted):
    h = fitted.holdout_validation(0.2)
    assert h["n_test"] > 0 and h["test_mape"] < 0.1


def test_unfitted_model_raises():
    cfg = MMMConfig(target_col="y", channel_cols=["a"])
    with pytest.raises(ModelNotFittedError):
        MarketingMixModel(cfg).channel_metrics()


def test_config_validation(synthetic):
    df, _ = synthetic
    with pytest.raises(KeyError):
        MarketingMixModel(MMMConfig(target_col="total_revenue", channel_cols=["nope"])).fit(df)
    with pytest.raises(ValueError):
        ChannelParams(alpha=1.2).validate()


# --------------------------------------------------------------------------- #
# Optimizer                                                                    #
# --------------------------------------------------------------------------- #
def test_optimizer_respects_budget_and_bounds(fitted):
    opt = BudgetOptimizer(fitted, horizon_weeks=52)
    budget = opt.historical_total
    res = opt.optimize(budget, OptimizerConstraints())
    a = res.allocation
    assert a["optimal_spend"].sum() == pytest.approx(budget, rel=1e-6)
    assert (a["optimal_spend"] >= a["lower_bound"] - 1e-3 * budget).all()
    assert (a["optimal_spend"] <= a["upper_bound"] + 1e-3 * budget).all()
    assert res.optimal_revenue >= res.current_revenue - 1e-6


def test_optimizer_equalises_marginal_roi(fitted):
    opt = BudgetOptimizer(fitted, horizon_weeks=52)
    res = opt.optimize(opt.historical_total * 1.5, OptimizerConstraints(min_share=0, max_share=1,
                                                                         min_mult=0, max_mult=10))
    interior = res.allocation[res.allocation["at_bound"] == "interior"]
    if len(interior) >= 2:
        mroi = interior["optimal_marginal_roi"]
        assert mroi.max() - mroi.min() < 0.05 * mroi.mean()


def test_optimizer_infeasible_and_relax(fitted):
    opt = BudgetOptimizer(fitted, horizon_weeks=52)
    tight = OptimizerConstraints(max_share=0.2)  # 4 channels x 20 % < 100 %
    with pytest.raises(InfeasibleConstraintsError):
        opt.optimize(opt.historical_total, tight, auto_relax=False)
    res = opt.optimize(opt.historical_total, tight, auto_relax=True)
    assert res.relaxations and res.allocation["optimal_spend"].sum() == pytest.approx(opt.historical_total)


def test_predict_revenue_matches_history(fitted):
    opt = BudgetOptimizer(fitted, horizon_weeks=52)
    assert opt.predict_revenue(opt.historical_spend.to_dict()) == pytest.approx(opt.historical_revenue, rel=1e-9)


# --------------------------------------------------------------------------- #
# Data layer                                                                   #
# --------------------------------------------------------------------------- #
def test_schema_detection_synthetic(synthetic):
    df, _ = synthetic
    g = detect_schema(df)
    assert g.date_col == "date"
    assert g.target_col == "total_revenue"
    assert set(g.channel_cols) == set(CHANNELS)
    assert set(g.control_cols) == set(CONTROLS)


def test_prepare_daily_messy_data():
    rng = np.random.default_rng(1)
    n = 120
    raw = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n).astype(str),
        "Facebook Spend": [f"${v:,.2f}" for v in rng.uniform(100, 2_000, n)],
        "TV": rng.uniform(0, 5_000, n),
        "Price": rng.uniform(9, 11, n),
        "Sales": rng.uniform(10_000, 20_000, n),
    })
    raw = coerce_numeric_columns(raw, exclude=["Date"])
    g = detect_schema(raw)
    assert g.date_col == "Date" and g.target_col == "Sales"
    assert set(g.channel_cols) == {"Facebook Spend", "TV"}
    data, notes = prepare_dataframe(raw, g.date_col, g.target_col, g.channel_cols, g.control_cols)
    assert len(data) < n                     # resampled to weekly
    assert any("weekly" in note for note in notes)
    assert pd.api.types.is_numeric_dtype(data["Facebook Spend"])
