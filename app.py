"""
app.py — Marketing Mix Modeling (MMM) & Budget Allocation Engine
=================================================================

Streamlit dashboard with three tabs:

1. Data Exploration & Overview   – spend trends, revenue, correlations, raw data
2. Model Fit & ROI Diagnostics   – fit statistics, coefficients, decomposition,
                                   ROI / marginal ROI, response & adstock curves
3. Budget Allocation Simulator   – live SciPy optimisation of a total budget
                                   under share-of-budget and baseline bounds

Run with:   streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the repo root importable however the app is launched (streamlit run, tests, IDE).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless backend – required on servers / Streamlit Cloud
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from data.generator import DEFAULT_TRUTH  # noqa: E402
from data.loader import (  # noqa: E402
    KAGGLE_HANDLE,
    DataLoadError,
    coerce_numeric_columns,
    detect_schema,
    load_kaggle_dataset,
    load_synthetic,
    prepare_dataframe,
    read_tabular_file,
)
from models.mmm import ChannelParams, MarketingMixModel, MMMConfig, adstock_half_life  # noqa: E402
from models.optimizer import BudgetOptimizer, InfeasibleConstraintsError, OptimizerConstraints  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mmm-app")

# =========================================================================== #
# Page config & visual system                                                  #
# =========================================================================== #
st.set_page_config(
    page_title="MMM & Budget Allocation Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Categorical palette in a fixed order (colour follows the channel, never its rank).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8984"          # base / "current" / non-media
NEUTRAL_LIGHT = "#d6d5d0"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
# Diverging map for correlations: orange (negative) ← gray → blue (positive)
DIVERGING = LinearSegmentedColormap.from_list("mmm_div", ["#d95926", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": NEUTRAL_LIGHT,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titleweight": "semibold",
    "axes.titlesize": 12,
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "font.family": "sans-serif",
})

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      .mmm-sub {color: #52514e; margin-top: -0.6rem; margin-bottom: 1rem;}
      .mmm-note {font-size: 0.85rem; color: #52514e;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================================== #
# Small helpers                                                                #
# =========================================================================== #
_ACRONYMS = {"tv", "ooh", "sem", "ppc", "crm", "ctv", "seo", "dm", "fb", "kpi", "gmv"}


def pretty(col: str) -> str:
    """'paid_search_spend' → 'Paid Search', 'tv_spend' → 'TV'."""
    s = str(col)
    for suffix in ("_spend", "_cost", "_budget", " spend", " Spend", "_Spend"):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)]
    words = s.replace("_", " ").split()
    return " ".join(w.upper() if w.lower() in _ACRONYMS else w.capitalize() for w in words) or str(col)


def money(x: float, decimals: int = 1) -> str:
    """Compact currency: $1.2M, $345.6K."""
    if x is None or not np.isfinite(x):
        return "–"
    sign = "-" if x < 0 else ""
    x = abs(x)
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{sign}${x / div:,.{decimals}f}{unit}"
    return f"{sign}${x:,.0f}"


def money_axis(ax, axis: str = "y", nbins: int = 5) -> None:
    """Compact $ tick labels with a capped number of ticks (avoids collisions)."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(mticker.MaxNLocator(nbins=nbins, min_n_ticks=3))
    target.set_major_formatter(mticker.FuncFormatter(lambda v, _: money(v, 1) if v else "$0"))


def date_axis(ax, max_ticks: int = 7) -> None:
    """Concise, non-overlapping date ticks."""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max_ticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def signed_money(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def channel_colors(channels: List[str]) -> Dict[str, str]:
    return {ch: (SERIES_COLORS[i] if i < len(SERIES_COLORS) else NEUTRAL) for i, ch in enumerate(channels)}


def show_fig(fig) -> None:
    """Render a matplotlib figure and free its memory."""
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def show_df(data, **kwargs) -> None:
    """st.dataframe at full width across Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:  # older Streamlit: width must be an int
        st.dataframe(data, use_container_width=True, **kwargs)


def df_fingerprint(df: pd.DataFrame) -> str:
    """Stable short hash of a DataFrame (used for widget keys)."""
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()[:10]


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode()


# =========================================================================== #
# Cached data & model functions                                                #
# =========================================================================== #
@st.cache_data(show_spinner="Generating synthetic data…")
def get_synthetic(seed: int, n_weeks: int) -> pd.DataFrame:
    if seed == 42 and n_weeks == 156:
        return load_synthetic()  # the committed / default CSV (matches ground-truth JSON)
    return load_synthetic(regenerate=True, seed=seed, n_weeks=n_weeks)


@st.cache_data(show_spinner="Downloading dataset from Kaggle…", ttl=24 * 3600)
def get_kaggle(file_path: str) -> Tuple[pd.DataFrame, str]:
    return load_kaggle_dataset(KAGGLE_HANDLE, file_path or None)


@st.cache_data(show_spinner="Reading file…")
def get_upload(content: bytes, name: str) -> pd.DataFrame:
    return read_tabular_file(io.BytesIO(content), name=name)


@st.cache_resource(show_spinner="Fitting the marketing-mix model…", max_entries=12)
def fit_model(
    data: pd.DataFrame,
    target: str,
    channels: Tuple[str, ...],
    controls: Tuple[str, ...],
    date_col: str,
    add_trend: bool,
    add_seasonality: bool,
    n_fourier: int,
    params: Tuple[Tuple[str, float, float, float], ...],
    auto_tune: bool,
) -> MarketingMixModel:
    cfg = MMMConfig(
        target_col=target,
        channel_cols=list(channels),
        control_cols=list(controls),
        date_col=date_col,
        add_trend=add_trend,
        add_seasonality=add_seasonality,
        n_fourier=n_fourier,
    )
    ch_params = {ch: ChannelParams(alpha=a, ec50_ratio=e, slope=s) for ch, a, e, s in params}
    return MarketingMixModel(cfg, ch_params).fit(data, auto_tune=auto_tune, n_passes=2)


def load_truth() -> Optional[dict]:
    try:
        return json.loads(DEFAULT_TRUTH.read_text())
    except Exception:
        return None


# =========================================================================== #
# Sidebar – data source                                                        #
# =========================================================================== #
st.sidebar.title("⚙️ Configuration")
st.sidebar.subheader("1 · Data source")
source = st.sidebar.radio(
    "Choose data",
    ["Synthetic demo (156 weeks)", f"Kaggle: {KAGGLE_HANDLE}", "Upload CSV / Excel"],
    help="The synthetic dataset has a known ground truth, so you can check that the model recovers it.",
)

raw_df: Optional[pd.DataFrame] = None
source_label = ""
is_synthetic = False
load_notes: List[str] = []

try:
    if source.startswith("Synthetic"):
        c1, c2 = st.sidebar.columns(2)
        seed = int(c1.number_input("Seed", value=42, min_value=0, max_value=10_000, step=1))
        n_weeks = int(c2.number_input("Weeks", value=156, min_value=60, max_value=520, step=4))
        raw_df = get_synthetic(seed, n_weeks)
        source_label = f"Synthetic data · seed {seed} · {n_weeks} weeks"
        is_synthetic = seed == 42 and n_weeks == 156
    elif source.startswith("Kaggle"):
        kfile = st.sidebar.text_input("File inside dataset (optional)", value="",
                                      help="Leave blank to use the largest CSV in the dataset.")
        try:
            raw_df, fname = get_kaggle(kfile.strip())
            source_label = f"Kaggle · {KAGGLE_HANDLE} · {fname}"
        except DataLoadError as exc:
            st.sidebar.error("Kaggle download failed – showing synthetic data instead.")
            load_notes.append(f"Kaggle load failed: {exc}")
            raw_df = get_synthetic(42, 156)
            source_label = "Synthetic data (Kaggle fallback)"
            is_synthetic = True
    else:
        up = st.sidebar.file_uploader("Upload weekly data", type=["csv", "tsv", "txt", "xlsx", "xls", "parquet", "json"])
        if up is None:
            st.title("📈 Marketing Mix Modeling & Budget Allocation Engine")
            st.info("Upload a CSV / Excel file with one row per week: a date, a revenue (or sales) "
                    "column, one spend column per channel and optional control variables.")
            st.stop()
        raw_df = get_upload(up.getvalue(), up.name)
        source_label = f"Upload · {up.name}"
except DataLoadError as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

if raw_df is None or raw_df.empty:
    st.error("The selected dataset is empty.")
    st.stop()

raw_df = coerce_numeric_columns(raw_df)
raw_key = df_fingerprint(raw_df)
guess = detect_schema(raw_df)

# --------------------------------------------------------------------------- #
# Sidebar – column mapping                                                     #
# --------------------------------------------------------------------------- #
st.sidebar.subheader("2 · Column mapping")
all_cols = list(raw_df.columns)
numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(raw_df[c])]
with st.sidebar.expander("Map columns", expanded=not is_synthetic):
    date_options = ["(none – use row order)"] + all_cols
    date_sel = st.selectbox("Date column", date_options,
                            index=date_options.index(guess.date_col) if guess.date_col in all_cols else 0,
                            key=f"date_{raw_key}")
    date_col_raw = None if date_sel.startswith("(none") else date_sel

    target_options = [c for c in numeric_cols]
    if not target_options:
        st.error("No numeric columns found in the data.")
        st.stop()
    target = st.selectbox("Target (revenue / sales)", target_options,
                          index=target_options.index(guess.target_col) if guess.target_col in target_options else 0,
                          key=f"target_{raw_key}")
    channel_options = [c for c in numeric_cols if c != target]
    channels = st.multiselect("Media spend channels", channel_options,
                              default=[c for c in guess.channel_cols if c in channel_options],
                              key=f"channels_{raw_key}")
    control_options = [c for c in numeric_cols if c not in (target, *channels)]
    controls = st.multiselect("Control variables", control_options,
                              default=[c for c in guess.control_cols if c in control_options],
                              key=f"controls_{raw_key}",
                              help="Non-media drivers entered linearly: macro indices, holidays, price, promotions…")

if not channels:
    st.title("📈 Marketing Mix Modeling & Budget Allocation Engine")
    st.warning("Select at least one media spend channel in the sidebar (Column mapping).")
    st.stop()

try:
    data, prep_notes = prepare_dataframe(raw_df, date_col_raw, target, channels, controls)
except DataLoadError as exc:
    st.error(f"Data preparation failed: {exc}")
    st.stop()
load_notes += guess.notes + prep_notes
date_col = date_col_raw or "date"
colors = channel_colors(channels)

# --------------------------------------------------------------------------- #
# Sidebar – model settings                                                     #
# --------------------------------------------------------------------------- #
st.sidebar.subheader("3 · Model settings")
auto_tune = st.sidebar.toggle("Auto-tune adstock & saturation", value=True,
                              help="Grid search (coordinate descent) over α, EC50 and slope that maximises "
                                   "adjusted R² while keeping media effects positive.")
c1, c2 = st.sidebar.columns(2)
add_trend = c1.checkbox("Trend", value=True)
add_seasonality = c2.checkbox("Seasonality", value=len(data) >= 52)
n_fourier = st.sidebar.slider("Fourier terms (yearly)", 1, 4, 2, disabled=not add_seasonality)

manual_params: List[Tuple[str, float, float, float]] = []
with st.sidebar.expander("Manual channel hyper-parameters", expanded=not auto_tune):
    st.caption("Used directly when auto-tune is off (and as the starting point when it is on).")
    for ch in channels:
        st.markdown(f"**{pretty(ch)}**")
        a = st.slider("Adstock decay α", 0.0, 0.95, 0.5, 0.05, key=f"a_{ch}_{raw_key}",
                      help="Share of last week's ad pressure that carries into this week.")
        e = st.slider("EC50 (× mean adstocked spend)", 0.1, 4.0, 1.0, 0.05, key=f"e_{ch}_{raw_key}",
                      help="Spend level at which the channel reaches half its maximum effect.")
        s = st.slider("Hill slope", 0.5, 4.0, 1.5, 0.1, key=f"s_{ch}_{raw_key}",
                      help="≤1: concave (diminishing returns). >1: S-curve (threshold effect).")
        manual_params.append((ch, a, e, s))

st.sidebar.markdown("---")
st.sidebar.caption("Adstock → Hill → OLS · SciPy SLSQP optimizer · statsmodels inference")

# =========================================================================== #
# Fit the model                                                                #
# =========================================================================== #
st.title("📈 Marketing Mix Modeling & Budget Allocation Engine")
st.markdown(f"<div class='mmm-sub'>{source_label} · {len(data)} periods · "
            f"{data[date_col].min():%d %b %Y} → {data[date_col].max():%d %b %Y}</div>",
            unsafe_allow_html=True)
if load_notes:
    with st.expander(f"ℹ️ Data notes ({len(load_notes)})"):
        for n in load_notes:
            st.markdown(f"- {n}")

try:
    model = fit_model(data, target, tuple(channels), tuple(controls), date_col, add_trend,
                      add_seasonality, int(n_fourier), tuple(manual_params), bool(auto_tune))
except Exception as exc:  # show any modelling error in the UI instead of a stack trace
    st.error(f"Model fitting failed: {exc}")
    st.stop()

metrics = model.fit_metrics()
ch_metrics = model.channel_metrics()
decomp = model.decompose()
model_key = f"{raw_key}_{hashlib.md5(str((target, channels, controls, auto_tune, manual_params, add_trend, add_seasonality, n_fourier)).encode()).hexdigest()[:8]}"

tab1, tab2, tab3 = st.tabs(["📊 Data Exploration & Overview", "🧮 Model Fit & ROI Diagnostics",
                            "💰 Budget Allocation Simulator"])

# =========================================================================== #
# TAB 1 – Data exploration                                                     #
# =========================================================================== #
with tab1:
    dates = pd.to_datetime(data[date_col])
    total_spend = float(data[channels].sum().sum())
    total_rev = float(data[target].sum())
    k = st.columns(5)
    k[0].metric("Periods", f"{len(data)}")
    k[1].metric("Total revenue", money(total_rev))
    k[2].metric("Total media spend", money(total_spend))
    k[3].metric("Spend / revenue", f"{total_spend / total_rev:.1%}" if total_rev else "–")
    k[4].metric("Channels · controls", f"{len(channels)} · {len(controls)}")

    # --- Revenue & total spend over time (two panels, one shared time axis) -
    st.subheader("Revenue and media spend over time")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.1, 1]})
    ax1.plot(dates, data[target], color=TEXT_PRIMARY, lw=1.8)
    ax1.set_title(f"{pretty(target)} per period")
    money_axis(ax1)
    hol_col = next((c for c in controls if "holiday" in c.lower()), None)
    if hol_col is not None:
        hol = data[hol_col] > 0
        ax1.scatter(dates[hol], data.loc[hol, target], s=36, color="#e34948", zorder=3,
                    edgecolor=SURFACE, linewidth=1.5, label="Holiday week")
        ax1.legend(loc="upper left")
    ax2.stackplot(dates, *[data[ch] for ch in channels], colors=[colors[c] for c in channels],
                  labels=[pretty(c) for c in channels], alpha=0.9, edgecolor=SURFACE, linewidth=0.6)
    ax2.set_title("Media spend by channel (stacked)")
    money_axis(ax2)
    ax2.legend(loc="upper left", ncol=min(len(channels), 4))
    date_axis(ax2, 10)
    show_fig(fig)

    # --- Spend trends per channel (small multiples) -------------------------
    st.subheader("Spend trends by channel")
    n = len(channels)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.6 * nrows), sharex=True, squeeze=False)
    for i, ch in enumerate(channels):
        ax = axes[i // ncols][i % ncols]
        ax.fill_between(dates, data[ch], color=colors[ch], alpha=0.15, linewidth=0)
        ax.plot(dates, data[ch], color=colors[ch], lw=1.6)
        roll = data[ch].rolling(8, min_periods=1).mean()
        ax.plot(dates, roll, color=TEXT_PRIMARY, lw=1.0, ls="--", label="8-period average")
        ax.set_title(f"{pretty(ch)} · total {money(data[ch].sum())}")
        money_axis(ax, nbins=4)
        date_axis(ax, 5)
        if i == 0:
            ax.legend(loc="upper left")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    show_fig(fig)

    # --- Correlation matrix + spend mix --------------------------------------
    c_left, c_right = st.columns([1.35, 1])
    with c_left:
        st.subheader("Correlation matrix")
        corr_cols = [target, *channels, *controls]
        corr = data[corr_cols].corr()
        corr.index = corr.columns = [pretty(c) for c in corr_cols]
        size = max(5.0, 0.75 * len(corr_cols) + 2)
        fig, ax = plt.subplots(figsize=(size, size * 0.8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1, center=0,
                    square=True, linewidths=2, linecolor=SURFACE, cbar_kws={"shrink": 0.75},
                    annot_kws={"size": 9}, ax=ax)
        ax.grid(False)
        ax.set_title("Pearson correlation (raw, untransformed)")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        show_fig(fig)
        st.caption("High correlation between spend channels (|r| > 0.7) makes their individual effects "
                   "hard to separate – a key MMM caveat.")
    with c_right:
        st.subheader("Spend mix")
        mix = data[channels].sum().sort_values()
        fig, ax = plt.subplots(figsize=(6, 0.6 * len(channels) + 1.6))
        bars = ax.barh([pretty(c) for c in mix.index], mix.values, color=[colors[c] for c in mix.index],
                       height=0.6)
        for b, v in zip(bars, mix.values):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {v / mix.sum():.0%}",
                    va="center", color=TEXT_SECONDARY, fontsize=9)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        money_axis(ax, "x")
        ax.set_xlim(0, mix.max() * 1.18)
        ax.set_title("Total spend by channel")
        show_fig(fig)

        st.subheader("Spend vs. revenue")
        sel = st.selectbox("Channel", channels, format_func=pretty, key=f"scatter_{raw_key}")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.scatter(data[sel], data[target], s=28, color=colors[sel], alpha=0.75, edgecolor=SURFACE, linewidth=1)
        ax.set_xlabel(f"{pretty(sel)} spend")
        ax.set_ylabel(pretty(target))
        money_axis(ax, "x")
        money_axis(ax, "y")
        ax.grid(axis="x")
        show_fig(fig)

    # --- Tables --------------------------------------------------------------
    st.subheader("Descriptive statistics")
    desc = data[[target, *channels, *controls]].describe().T
    desc.index = [pretty(c) for c in desc.index]
    show_df(desc.style.format("{:,.2f}"))
    with st.expander("Raw (prepared) data"):
        show_df(data, hide_index=True)
        st.download_button("⬇️ Download prepared data (CSV)", data.to_csv(index=False).encode(),
                           file_name="mmm_prepared_data.csv", mime="text/csv")

# =========================================================================== #
# TAB 2 – Model fit & ROI diagnostics                                          #
# =========================================================================== #
with tab2:
    for w in model.warnings_:
        st.warning(w, icon="⚠️")

    try:
        holdout = model.holdout_validation(0.2)
    except Exception as exc:
        holdout = None
        st.info(f"Hold-out validation skipped: {exc}")

    k = st.columns(6)
    k[0].metric("R²", f"{metrics['r2']:.3f}")
    k[1].metric("Adjusted R²", f"{metrics['adj_r2']:.3f}")
    k[2].metric("MAPE (in-sample)", f"{metrics['mape']:.2%}")
    k[3].metric("MAPE (hold-out)", f"{holdout['test_mape']:.2%}" if holdout else "–",
                help="Coefficients re-estimated on the first 80 % of periods, scored on the last 20 %.")
    k[4].metric("Durbin-Watson", f"{metrics['durbin_watson']:.2f}", help="≈2 means no residual autocorrelation.")
    k[5].metric("F-test p-value", f"{metrics['f_pvalue']:.1e}")

    # --- Actual vs fitted ----------------------------------------------------
    st.subheader("Actual vs. fitted")
    dts = pd.to_datetime(decomp[date_col])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.plot(dts, decomp["actual"], color=TEXT_PRIMARY, lw=1.6, label="Actual")
    ax1.plot(dts, decomp["fitted"], color=SERIES_COLORS[0], lw=2.0, label="Fitted")
    if holdout:
        split = dts.iloc[holdout["n_train"]]
        ax1.axvline(split, color=NEUTRAL, ls=":", lw=1.2)
        ax1.text(split, ax1.get_ylim()[1], "  hold-out →", va="top", color=TEXT_SECONDARY, fontsize=9)
    ax1.set_title(f"{pretty(target)}: actual vs. model")
    ax1.legend(loc="upper left", ncol=2)
    money_axis(ax1)
    resid = decomp["actual"] - decomp["fitted"]
    ax2.bar(dts, resid, width=5, color=np.where(resid >= 0, SERIES_COLORS[0], SERIES_COLORS[1]))
    ax2.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax2.set_title("Residuals")
    money_axis(ax2, nbins=4)
    date_axis(ax2, 10)
    show_fig(fig)

    # --- Channel ROI table -----------------------------------------------------
    st.subheader("Channel ROI & contribution")
    tbl = ch_metrics.copy()
    tbl.index = [pretty(c) for c in tbl.index]
    tbl = tbl[["spend", "spend_share", "contribution", "contribution_pct_revenue", "contribution_pct_media",
               "roi", "marginal_roi", "p_value", "alpha", "half_life_weeks", "ec50_ratio", "slope"]]
    tbl.columns = ["Spend", "Spend share", "Contribution", "% of revenue", "% of media revenue",
                   "ROI", "Marginal ROI", "p-value", "Adstock α", "Half-life (periods)", "EC50 ratio", "Hill slope"]
    show_df(tbl.style.format({
        "Spend": "${:,.0f}", "Contribution": "${:,.0f}", "Spend share": "{:.1%}", "% of revenue": "{:.1%}",
        "% of media revenue": "{:.1%}", "ROI": "{:.2f}", "Marginal ROI": "{:.2f}", "p-value": "{:.4f}",
        "Adstock α": "{:.2f}", "Half-life (periods)": "{:.1f}", "EC50 ratio": "{:.2f}", "Hill slope": "{:.2f}",
    }))
    st.markdown("<div class='mmm-note'><b>ROI</b> = revenue attributed to the channel ÷ its spend "
                "(average return of every dollar). <b>Marginal ROI</b> = revenue from the <i>next</i> dollar "
                "at current spend – the number that should drive reallocation. A marginal ROI below 1.0 means "
                "the last dollar did not pay for itself.</div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)
    with c_left:
        # --- ROI vs marginal ROI (grouped bars) ----------------------------------
        fig, ax = plt.subplots(figsize=(6.4, 4))
        x = np.arange(len(channels))
        w = 0.38
        ax.bar(x - w / 2, ch_metrics["roi"], w, color=[colors[c] for c in channels], label="ROI")
        ax.bar(x + w / 2, ch_metrics["marginal_roi"], w, color=[colors[c] for c in channels], alpha=0.45,
               hatch="///", edgecolor=SURFACE, label="Marginal ROI")
        ax.axhline(1.0, color=TEXT_SECONDARY, ls="--", lw=1)
        ax.text(len(channels) - 0.5, 1.0, "break-even", va="bottom", ha="right", color=TEXT_SECONDARY, fontsize=8)
        ax.set_xticks(x, [pretty(c) for c in channels], rotation=20, ha="right")
        ax.set_ylabel("Revenue per $1 of spend")
        ax.set_title("ROI vs. marginal ROI")
        ax.legend(loc="upper right")
        show_fig(fig)
    with c_right:
        # --- Contribution breakdown ----------------------------------------------
        totals = decomp.drop(columns=[date_col, "fitted", "actual"]).sum()
        order = [MarketingMixModel.BASE_COMPONENT, *[c for c in controls if c in totals.index], *channels]
        totals = totals[order]
        fig, ax = plt.subplots(figsize=(6.4, 4))
        def _component_label(c: str) -> str:
            if c == MarketingMixModel.BASE_COMPONENT:
                return "Base"
            if model.control_center_.get(c, 0.0) != 0.0:
                return f"{pretty(c)} (vs. avg)"  # centred control: effect of deviations
            return pretty(c)

        labels = [_component_label(c) for c in order]
        cols = [NEUTRAL if c not in channels else colors[c] for c in order]
        cols = [NEUTRAL_LIGHT if c in controls else col for c, col in zip(order, cols)]
        ax.barh(labels[::-1], totals.values[::-1], color=cols[::-1], height=0.6)
        share = totals / decomp["fitted"].sum()
        for i, (v, s_) in enumerate(zip(totals.values[::-1], share.values[::-1])):
            ax.text(v, i, f"  {s_:.1%}" if v >= 0 else f"{s_:.1%}  ", va="center",
                    ha="left" if v >= 0 else "right", color=TEXT_SECONDARY, fontsize=9)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        money_axis(ax, "x")
        ax.set_xlim(min(0, totals.min() * 1.3), totals.max() * 1.22)
        ax.set_title("Revenue decomposition (total over period)")
        show_fig(fig)

    # --- Media contribution over time -------------------------------------------
    st.subheader("Media-driven revenue over time")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    contrib_stack = [decomp[ch].clip(lower=0) for ch in channels]
    ax.stackplot(dts, *contrib_stack, colors=[colors[c] for c in channels], labels=[pretty(c) for c in channels],
                 alpha=0.9, edgecolor=SURFACE, linewidth=0.6)
    ax.set_title("Weekly revenue contribution by channel (stacked)")
    ax.legend(loc="upper left", ncol=min(len(channels), 4))
    money_axis(ax)
    date_axis(ax, 10)
    show_fig(fig)

    # --- Response curves & adstock decay --------------------------------------------
    c_left, c_right = st.columns([1.5, 1])
    with c_left:
        st.subheader("Response curves (saturation)")
        n = len(channels)
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8, 2.7 * nrows), squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncols][i % ncols]
            rc = model.response_curve(ch, np.linspace(0, 3, 61))
            ax.plot(rc["spend"], rc["contribution"], color=colors[ch], lw=2)
            cur = rc.loc[(rc["multiplier"] - 1.0).abs().idxmin()]
            ax.scatter([cur["spend"]], [cur["contribution"]], s=60, color=colors[ch], edgecolor=SURFACE,
                       linewidth=2, zorder=3)
            ax.annotate("current", (cur["spend"], cur["contribution"]), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color=TEXT_SECONDARY)
            ax.set_title(pretty(ch))
            money_axis(ax, "x", nbins=4)
            money_axis(ax, "y", nbins=4)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.supxlabel("Total period spend (historical pattern scaled 0–3×)", fontsize=9, color=TEXT_SECONDARY)
        fig.supylabel("Attributed revenue", fontsize=9, color=TEXT_SECONDARY)
        show_fig(fig)
    with c_right:
        st.subheader("Adstock decay")
        fig, ax = plt.subplots(figsize=(5.5, 3.9))
        lags = np.arange(0, 13)
        for ch in channels:
            a = model.channel_params[ch].alpha
            ax.plot(lags, a ** lags, color=colors[ch], marker="o", ms=4,
                    label=f"{pretty(ch)} (α={a:.2f}, t½={adstock_half_life(a):.1f})")
        ax.set_xlabel("Periods after spend")
        ax.set_ylabel("Share of effect remaining")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title("How long each channel is remembered")
        ax.legend(loc="upper right", fontsize=8)
        show_fig(fig)

    # --- Coefficients ----------------------------------------------------------
    st.subheader("Regression coefficients")
    coef = model.coefficient_table().copy()
    coef.index = [pretty(c) if c in channels or c in controls else c for c in coef.index]

    def _sig_style(row):
        if row["type"] == "structural":
            return [""] * len(row)
        color = "background-color: rgba(27,175,122,0.12)" if row["p_value"] < 0.05 else \
            "background-color: rgba(235,104,52,0.12)"
        return [color] * len(row)

    show_df(coef.style.apply(_sig_style, axis=1).format({
        "coefficient": "{:,.1f}", "std_error": "{:,.1f}", "t_stat": "{:.2f}", "p_value": "{:.4f}",
        "ci_lower": "{:,.1f}", "ci_upper": "{:,.1f}", "vif": "{:.2f}",
    }))
    st.caption("Media coefficients = maximum weekly revenue at full saturation (the Hill output is in [0, 1)). "
               "Green rows are significant at 5 %, orange are not. VIF > 10 signals multicollinearity.")

    with st.expander("Full statsmodels OLS summary"):
        st.code(model.summary_text(), language="text")
    if model.tuning_log_:
        with st.expander("Hyper-parameter tuning log"):
            show_df(pd.DataFrame(model.tuning_log_), hide_index=True)

    # --- Ground-truth check (synthetic data only) ----------------------------
    truth = load_truth() if is_synthetic else None
    if truth and set(channels) <= set(truth["channels"]):
        with st.expander("✅ Ground-truth recovery check (synthetic data)", expanded=False):
            rows = []
            for ch in channels:
                t = truth["channels"][ch]
                rows.append({"Channel": pretty(ch), "True ROI": t["true_roi"], "Estimated ROI": ch_metrics.loc[ch, "roi"],
                             "True α": t["alpha"], "Estimated α": model.channel_params[ch].alpha,
                             "True % of revenue": t["true_contribution_pct_revenue"],
                             "Estimated % of revenue": ch_metrics.loc[ch, "contribution_pct_revenue"]})
            show_df(pd.DataFrame(rows).set_index("Channel").style.format({
                "True ROI": "{:.2f}", "Estimated ROI": "{:.2f}", "True α": "{:.2f}", "Estimated α": "{:.2f}",
                "True % of revenue": "{:.1%}", "Estimated % of revenue": "{:.1%}"}))
            st.caption("The synthetic data were generated with known adstock, saturation and effect sizes. "
                       "Perfect recovery is not expected – always-on channels are partly confounded with the "
                       "baseline – but the ranking and orders of magnitude should match.")

    st.download_button("⬇️ Download channel metrics (CSV)", csv_bytes(ch_metrics), file_name="mmm_channel_metrics.csv",
                       mime="text/csv")

# =========================================================================== #
# TAB 3 – Budget allocation simulator                                          #
# =========================================================================== #
with tab3:
    st.markdown("Re-allocate a total budget across channels to maximise predicted revenue. Each channel keeps "
                "its historical weekly flighting pattern; the optimizer (SciPy SLSQP, multi-start) decides how "
                "many dollars each channel gets.")

    n_periods = len(data)
    horizon_opts = {f"Last {h} periods": h for h in (13, 26, 52) if h < n_periods}
    horizon_opts[f"Full history ({n_periods} periods)"] = n_periods
    default_h = "Last 52 periods" if "Last 52 periods" in horizon_opts else list(horizon_opts)[-1]

    c1, c2 = st.columns([1, 2])
    with c1:
        horizon_label = st.selectbox("Planning horizon", list(horizon_opts), index=list(horizon_opts).index(default_h),
                                     help="The budget covers this many periods, using their historical weekly pattern.")
    horizon = horizon_opts[horizon_label]
    try:
        optimizer = BudgetOptimizer(model, horizon_weeks=horizon)
    except Exception as exc:
        st.error(f"Could not initialise the optimizer: {exc}")
        st.stop()
    hist_total = optimizer.historical_total
    step = float(10 ** max(int(np.floor(np.log10(max(hist_total, 1)))) - 2, 0))
    with c2:
        budget = st.slider(
            f"Total budget for the horizon (historical: {money(hist_total, 2)})",
            min_value=float(np.floor(0.5 * hist_total / step) * step),
            max_value=float(np.ceil(2.0 * hist_total / step) * step),
            value=float(round(hist_total / step) * step),
            step=step, format="$%.0f", key=f"budget_{model_key}_{horizon}",
        )
    rel = budget / hist_total - 1 if hist_total else 0.0
    rel_txt = "same as" if abs(rel) < 0.0005 else f"{rel:+.1%} vs."
    st.caption(f"Budget = **{money(budget, 2)}** · {rel_txt} historical spend over the horizon")

    with st.expander("🎚️ Constraints", expanded=True):
        cc1, cc2, cc3 = st.columns(3)
        share_lo, share_hi = cc1.slider("Share of total budget per channel", 0, 100, (5, 50), 1, format="%d%%",
                                        help="No channel may receive less / more than this share of the budget.")
        mult_lo, mult_hi = cc2.slider("Change vs. baseline per channel", 0.0, 3.0, (0.5, 1.5), 0.05, format="%.2f×",
                                      help="0.50×–1.50× = each channel may move at most ±50 % from its baseline.")
        with cc3:
            scale_base = st.toggle("Baseline = historical mix at this budget", value=True,
                                   help="On: bounds are relative to the historical mix scaled to the new budget. "
                                        "Off: relative to historical dollars.")
            auto_relax = st.toggle("Auto-relax infeasible bounds", value=True)

        st.markdown("**Per-channel overrides** (multipliers of baseline)")
        override_df = pd.DataFrame({"Channel": [pretty(c) for c in channels],
                                    "Min ×": [mult_lo] * len(channels), "Max ×": [mult_hi] * len(channels)})
        try:
            edited = st.data_editor(
                override_df, hide_index=True, key=f"ovr_{model_key}_{mult_lo}_{mult_hi}",
                disabled=["Channel"],
                column_config={"Min ×": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.05, format="%.2f"),
                               "Max ×": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.05, format="%.2f")},
            )
        except Exception:
            edited = override_df
        overrides = {}
        for ch, (_, row) in zip(channels, edited.iterrows()):
            lo_m, hi_m = float(row["Min ×"]), float(row["Max ×"])
            if lo_m > hi_m:
                st.warning(f"{pretty(ch)}: Min × is above Max × – values swapped.")
                lo_m, hi_m = hi_m, lo_m
            overrides[ch] = (lo_m, hi_m)

    constraints = OptimizerConstraints(
        min_share=share_lo / 100, max_share=share_hi / 100, min_mult=mult_lo, max_mult=mult_hi,
        channel_mult_bounds=overrides, scale_baseline_to_budget=scale_base,
    )

    try:
        with st.spinner("Optimising allocation…"):
            result = optimizer.optimize(budget, constraints, n_random_starts=4, auto_relax=auto_relax)
    except InfeasibleConstraintsError as exc:
        st.error(f"{exc}\n\nWiden the share or baseline limits, or enable auto-relax.")
        st.stop()
    except Exception as exc:
        st.error(f"Optimisation failed: {exc}")
        st.stop()

    for r in result.relaxations:
        st.warning(r, icon="🔧")
    if not result.success:
        st.info(f"Solver note: {result.message} – showing the best feasible allocation found.")
    neg = [c for c in channels if model.results_.params[c] < 0]
    if neg:
        st.warning(f"Channels with negative estimated effect are pushed to their minimum: {', '.join(map(pretty, neg))}")

    alloc = result.allocation
    k = st.columns(4)
    k[0].metric("Revenue · current mix", money(result.current_revenue, 2),
                help="Historical channel mix applied to the chosen budget.")
    k[1].metric("Revenue · optimal mix", money(result.optimal_revenue, 2),
                delta=f"{money(result.lift_abs, 2)} ({result.lift_pct:+.2%})")
    media_lift = (result.optimal_media_revenue / result.current_media_revenue - 1) if result.current_media_revenue else np.nan
    k[2].metric("Media-driven revenue lift", f"{media_lift:+.1%}",
                help="Change in revenue attributed to media only (excludes the unaffected baseline).")
    k[3].metric("Media ROI", f"{result.optimal_roi:.2f}", delta=f"{result.optimal_roi - result.current_roi:+.2f} vs current")

    # --- Current vs optimal spend & contribution -------------------------------
    c_left, c_right = st.columns(2)
    labels = [pretty(c) for c in alloc.index]
    y = np.arange(len(labels))
    h = 0.38
    with c_left:
        fig, ax = plt.subplots(figsize=(6.4, 0.75 * len(labels) + 1.8))
        ax.barh(y + h / 2, alloc["current_spend"], h, color=NEUTRAL_LIGHT, label="Current mix")
        ax.barh(y - h / 2, alloc["optimal_spend"], h, color=[colors[c] for c in alloc.index], label="Optimal")
        for i, (lo_, hi_) in enumerate(zip(alloc["lower_bound"], alloc["upper_bound"])):
            ax.plot([lo_, hi_], [i - h - 0.08] * 2, color=TEXT_SECONDARY, lw=1)
            ax.plot([lo_, lo_], [i - h - 0.14, i - h - 0.02], color=TEXT_SECONDARY, lw=1)
            ax.plot([hi_, hi_], [i - h - 0.14, i - h - 0.02], color=TEXT_SECONDARY, lw=1)
        for i, pct in enumerate(alloc["spend_change_pct"]):
            ax.text(alloc["optimal_spend"].iloc[i], i - h / 2, f"  {pct:+.0%}", va="center", fontsize=9,
                    color=TEXT_PRIMARY)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        money_axis(ax, "x")
        ax.set_xlim(0, max(alloc["upper_bound"].max(), alloc["optimal_spend"].max()) * 1.15)
        ax.set_title("Spend: current vs. optimal (whiskers = allowed range)")
        ax.legend(handles=[Patch(color=NEUTRAL_LIGHT, label="Current mix"),
                           Patch(color=TEXT_SECONDARY, label="Optimal (channel colour)")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        show_fig(fig)
    with c_right:
        fig, ax = plt.subplots(figsize=(6.4, 0.75 * len(labels) + 1.8))
        ax.barh(y + h / 2, alloc["current_contribution"], h, color=NEUTRAL_LIGHT, label="Current mix")
        ax.barh(y - h / 2, alloc["optimal_contribution"], h, color=[colors[c] for c in alloc.index], label="Optimal")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        money_axis(ax, "x")
        ax.set_title("Attributed revenue: current vs. optimal")
        ax.legend(handles=[Patch(color=NEUTRAL_LIGHT, label="Current mix"),
                           Patch(color=TEXT_SECONDARY, label="Optimal (channel colour)")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        show_fig(fig)

    # --- Response curves with current & optimal points --------------------------
    st.subheader("Where each channel sits on its response curve")
    # Extend the curves far enough to show every channel's upper bound.
    hist_pos = alloc["historical_spend"].replace(0, np.nan)
    reach = float((alloc["upper_bound"] / hist_pos).max(skipna=True)) if hist_pos.notna().any() else 2.5
    curves = optimizer.response_curves(max_mult=float(np.clip(1.2 * reach, 2.0, 6.0)))
    n = len(channels)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.9 * nrows), squeeze=False)
    for i, ch in enumerate(channels):
        ax = axes[i // ncols][i % ncols]
        cdf = curves[curves["channel"] == ch]
        ax.plot(cdf["spend"], cdf["contribution"], color=colors[ch], lw=2)
        ax.axvspan(alloc.loc[ch, "lower_bound"], alloc.loc[ch, "upper_bound"], color=colors[ch], alpha=0.08, lw=0)
        ax.scatter([alloc.loc[ch, "current_spend"]], [alloc.loc[ch, "current_contribution"]], s=55,
                   color=NEUTRAL, edgecolor=SURFACE, linewidth=2, zorder=3, label="Current")
        ax.scatter([alloc.loc[ch, "optimal_spend"]], [alloc.loc[ch, "optimal_contribution"]], s=70,
                   color=colors[ch], edgecolor=SURFACE, linewidth=2, zorder=4, label="Optimal")
        ax.set_title(f"{pretty(ch)} · mROI {alloc.loc[ch, 'optimal_marginal_roi']:.2f}")
        money_axis(ax, "x", nbins=3)
        money_axis(ax, "y", nbins=4)
        if i == 0:
            ax.legend(loc="lower right", fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    show_fig(fig)
    st.caption("Shaded band = allowed spend range. At the optimum, channels that are not at a bound share the "
               "same marginal ROI – moving a dollar between them would not raise revenue.")

    # --- Allocation table ---------------------------------------------------------
    st.subheader("Allocation table")
    out = alloc[["historical_spend", "current_spend", "optimal_spend", "spend_change", "spend_change_pct",
                 "optimal_share", "current_roi", "optimal_roi", "current_marginal_roi", "optimal_marginal_roi",
                 "lower_bound", "upper_bound", "at_bound"]].copy()
    out.index = [pretty(c) for c in out.index]
    out.columns = ["Historical spend", "Current-mix spend", "Optimal spend", "Δ spend", "Δ %", "Optimal share",
                   "ROI (current)", "ROI (optimal)", "mROI (current)", "mROI (optimal)", "Min allowed",
                   "Max allowed", "Binding bound"]
    money_cols = ["Historical spend", "Current-mix spend", "Optimal spend", "Δ spend", "Min allowed", "Max allowed"]
    show_df(out.style.format({**{c: signed_money for c in money_cols}, "Δ %": "{:+.1%}", "Optimal share": "{:.1%}",
                              "ROI (current)": "{:.2f}", "ROI (optimal)": "{:.2f}",
                              "mROI (current)": "{:.2f}", "mROI (optimal)": "{:.2f}"}))
    st.download_button("⬇️ Download allocation plan (CSV)", csv_bytes(alloc), file_name="mmm_optimal_allocation.csv",
                       mime="text/csv")

    # --- Budget frontier ------------------------------------------------------------
    with st.expander("📈 Budget frontier – how revenue scales with total budget"):
        st.caption("Optimal vs. current-mix media revenue across total budgets from 0.5× to 2× historical, "
                   "using the constraints above (auto-relaxed where needed).")
        if st.button("Compute frontier", key=f"frontier_{model_key}_{horizon}"):
            with st.spinner("Running the optimizer across budget levels…"):
                sweep = optimizer.budget_sweep(np.linspace(0.5, 2.0, 7), constraints)
            fig, ax = plt.subplots(figsize=(12, 3.8))
            ax.plot(sweep["total_budget"], sweep["current_media_revenue"], color=NEUTRAL, marker="o", ms=5,
                    label="Current mix")
            ax.plot(sweep["total_budget"], sweep["optimal_media_revenue"], color=SERIES_COLORS[0], marker="o", ms=5,
                    label="Optimal mix")
            ax.axvline(hist_total, color=TEXT_SECONDARY, ls=":", lw=1)
            ax.text(hist_total, ax.get_ylim()[0], "  historical budget", va="bottom", fontsize=8, color=TEXT_SECONDARY)
            ax.set_xlabel("Total media budget for the horizon")
            ax.set_ylabel("Media-driven revenue")
            money_axis(ax, "x")
            money_axis(ax, "y")
            ax.grid(axis="x")
            ax.legend(loc="upper left")
            ax.set_title("Media revenue frontier")
            show_fig(fig)
            sweep_disp = sweep.copy()
            sweep_disp["incremental ROI of last step"] = sweep["optimal_media_revenue"].diff() / sweep["total_budget"].diff()
            show_df(sweep_disp.style.format({"budget_multiplier": "{:.2f}×", "total_budget": "${:,.0f}",
                                             "current_revenue": "${:,.0f}", "optimal_revenue": "${:,.0f}",
                                             "current_media_revenue": "${:,.0f}", "optimal_media_revenue": "${:,.0f}",
                                             "incremental ROI of last step": "{:.2f}"}, na_rep="–"), hide_index=True)

    st.markdown("<div class='mmm-note'>Assumptions: response curves are held fixed at their estimated shape; "
                "non-media drivers (base, trend, seasonality, controls) are unchanged; carry-over from spend "
                "<i>after</i> the horizon is not counted. Treat the output as directional guidance and validate "
                "large shifts with incrementality tests.</div>", unsafe_allow_html=True)
