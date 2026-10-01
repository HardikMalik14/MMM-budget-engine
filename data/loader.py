"""
data/loader.py
==============

Data access layer: synthetic demo data, the Kaggle ``nafees2006/mmm-dataset``
(via ``kagglehub``), or any user-uploaded CSV / Excel file.

Real-world MMM files never share a schema, so this module also contains a
heuristic *schema detector* that guesses the date, target, media-spend and
control columns. The dashboard shows the guess and lets the user override it.

Main functions
--------------
- ``load_synthetic()``         – read (or generate) the synthetic CSV
- ``load_kaggle_dataset()``    – download + read the Kaggle dataset
- ``read_tabular_file()``      – read an uploaded CSV / Excel / Parquet / JSON
- ``detect_schema()``          – guess column roles
- ``prepare_dataframe()``      – clean, coerce, sort and (optionally) resample to weekly
"""

from __future__ import annotations

import io
import logging
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.generator import DEFAULT_CSV, DEFAULT_TRUTH, save_synthetic_data  # noqa: E402

logger = logging.getLogger(__name__)

KAGGLE_HANDLE = "nafees2006/mmm-dataset"
TABULAR_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json")


class DataLoadError(RuntimeError):
    """Raised when a data source cannot be loaded or understood."""


# =========================================================================== #
# Schema detection                                                             #
# =========================================================================== #
@dataclass
class SchemaGuess:
    """Best guess of each column's role in an MMM dataset."""

    date_col: Optional[str] = None
    target_col: Optional[str] = None
    channel_cols: List[str] = field(default_factory=list)
    control_cols: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# Regexes are matched against a normalised (lower-case, '_' separated) name.
_DATE_PAT = re.compile(r"(^|_)(date|week|wk|day|time|period|month|ds|calendar)(_|$)")
_TARGET_PATS = [  # in priority order
    re.compile(r"(total_)?revenue"),
    re.compile(r"(total_)?sales"),
    re.compile(r"gmv|turnover|income"),
    re.compile(r"conversions?|orders?|units(_sold)?|kpi|target|^y$"),
]
_SPEND_PAT = re.compile(r"spend|cost|budget|invest|expense|media|adspend")
_CHANNEL_PAT = re.compile(
    r"\b(tv|television|radio|newspaper|print|magazine|ooh|outdoor|billboard|search|sem|ppc|google|"
    r"bing|social|facebook|fb|meta|instagram|tiktok|youtube|snap|linkedin|twitter|display|banner|"
    r"programmatic|video|influencer|affiliate|email|crm|podcast|ctv|digital|online|sponsorship)\b"
)
_NON_MEDIA_PAT = re.compile(
    r"price|discount|promo|holiday|macro|index|competitor|temperature|weather|gdp|cpi|"
    r"unemploy|event|trend|season|flag|dummy|stock|inventory|distribution|covid"
)
_ID_PAT = re.compile(
    r"^(id|index|unnamed.*|row|row_?num|week|week_?num(ber)?|week_?of_?year|year|month|month_?num|day|t)$|_id$"
)


def _norm(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return s


def _looks_like_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).head(50)
        if sample.empty:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "could not infer format" noise
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.9
    return False


def detect_schema(df: pd.DataFrame) -> SchemaGuess:
    """Heuristically assign roles to columns.

    Rules (first match wins):
    1. **Date** – datetime dtype, or a name like *date/week/period* whose values parse as dates.
    2. **Target** – revenue > sales > GMV > conversions/orders/KPI.
    3. **Channels** – numeric, non-negative columns whose names mention spend/cost or a
       known media channel (TV, radio, search, social, ...), excluding obvious controls.
    4. **Controls** – remaining numeric columns that are not IDs / counters.
    """
    guess = SchemaGuess()
    cols = list(df.columns)
    normed = {c: _norm(c) for c in cols}

    # 1) date
    date_candidates = [c for c in cols if _looks_like_date(df[c])]
    # Prefer columns whose *name* also says date/week/period.
    date_candidates.sort(key=lambda c: _DATE_PAT.search(normed[c]) is None)
    if date_candidates:
        guess.date_col = date_candidates[0]
    if guess.date_col is None:
        guess.notes.append("No date column detected – rows will be treated as consecutive weeks.")

    numeric = [c for c in cols if c != guess.date_col and pd.api.types.is_numeric_dtype(df[c])]

    # 2) target
    for pat in _TARGET_PATS:
        hits = [c for c in numeric if pat.search(normed[c])]
        if hits:
            # Prefer a column with "total" in it, then the one with the largest mean.
            hits.sort(key=lambda c: ("total" not in normed[c], -float(df[c].mean())))
            guess.target_col = hits[0]
            break
    if guess.target_col is None and numeric:
        guess.target_col = numeric[-1]
        guess.notes.append(f"No obvious revenue/sales column – defaulting target to '{numeric[-1]}'.")

    # 3) channels
    for c in numeric:
        if c == guess.target_col:
            continue
        n = normed[c]
        if _ID_PAT.search(n) or _NON_MEDIA_PAT.search(n):
            continue
        if (_SPEND_PAT.search(n) or _CHANNEL_PAT.search(n.replace("_", " "))) and float(df[c].min()) >= 0:
            guess.channel_cols.append(c)

    # Prefer *_spend columns if both "tv_spend" and "tv_impressions" exist.
    spend_like = [c for c in guess.channel_cols if _SPEND_PAT.search(normed[c])]
    if spend_like and len(spend_like) >= 2:
        dropped = [c for c in guess.channel_cols if c not in spend_like]
        if dropped:
            guess.notes.append(f"Non-spend media metrics not used as channels: {dropped}")
        guess.channel_cols = spend_like

    # 4) controls
    for c in numeric:
        if c in (guess.target_col, *guess.channel_cols):
            continue
        if _ID_PAT.search(normed[c]):
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        guess.control_cols.append(c)

    if not guess.channel_cols:
        guess.notes.append("No media spend columns detected – please select them manually.")
    return guess


# =========================================================================== #
# Cleaning                                                                     #
# =========================================================================== #
def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Turn '$1,234.50', '12%' or ' 1 200 ' style strings into floats."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    cleaned = (series.astype(str)
               .str.replace(r"[\$€£₹,%\s]", "", regex=True)
               .str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (123) -> -123
               .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan}))
    return pd.to_numeric(cleaned, errors="coerce")


def coerce_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None,
                           min_valid_share: float = 0.8) -> pd.DataFrame:
    """Convert object columns that are *mostly* numbers into float columns."""
    out = df.copy()
    exclude = set(exclude or [])
    for c in out.columns:
        if c in exclude or pd.api.types.is_numeric_dtype(out[c]) or pd.api.types.is_datetime64_any_dtype(out[c]):
            continue
        converted = _coerce_numeric(out[c])
        if converted.notna().mean() >= min_valid_share:
            out[c] = converted
    return out


def prepare_dataframe(
    df: pd.DataFrame,
    date_col: Optional[str],
    target_col: str,
    channel_cols: List[str],
    control_cols: List[str],
    resample_weekly: Union[bool, str] = "auto",
) -> Tuple[pd.DataFrame, List[str]]:
    """Clean a raw frame into a model-ready weekly frame.

    Steps: keep only the needed columns → coerce numbers → parse & sort dates
    (or create a synthetic weekly index) → aggregate duplicate dates → resample
    daily data to weekly (spend & target summed, controls averaged, 0/1 flags
    max-ed) → fill missing spend with 0.

    Returns
    -------
    (DataFrame, list[str])
        Clean data (with a ``date`` column named as ``date_col`` or ``"date"``)
        and a list of human-readable notes about what was changed.
    """
    notes: List[str] = []
    if not channel_cols:
        raise DataLoadError("Select at least one media spend column.")
    if target_col is None:
        raise DataLoadError("Select a target (revenue / sales) column.")
    keep = [c for c in [date_col, target_col, *channel_cols, *control_cols] if c]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise DataLoadError(f"Columns not found: {missing}")
    data = df[keep].copy()
    data = coerce_numeric_columns(data, exclude=[date_col] if date_col else [])

    for c in [target_col, *channel_cols, *control_cols]:
        if not pd.api.types.is_numeric_dtype(data[c]):
            data[c] = _coerce_numeric(data[c])
        bad = int(data[c].isna().sum())
        if bad and c in channel_cols:
            notes.append(f"'{c}': {bad} non-numeric/missing spend values set to 0.")
            data[c] = data[c].fillna(0.0)

    out_date = date_col or "date"
    if date_col:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        n_bad = int(data[date_col].isna().sum())
        if n_bad:
            notes.append(f"Dropped {n_bad} rows with unparseable dates.")
            data = data.dropna(subset=[date_col])
        data = data.sort_values(date_col)

        # Duplicate dates (e.g. one row per region) → aggregate.
        if data[date_col].duplicated().any():
            notes.append("Duplicate dates found (multiple rows per period) – aggregated to one row per date.")
            agg = {c: "sum" for c in [target_col, *channel_cols]}
            agg.update({c: "mean" for c in control_cols})
            data = data.groupby(date_col, as_index=False).agg(agg)

        # Daily → weekly
        if len(data) > 2:
            median_gap = data[date_col].diff().dt.days.median()
            need = (resample_weekly is True) or (resample_weekly == "auto" and median_gap is not None
                                                 and median_gap < 6)
            if need:
                agg = {c: "sum" for c in [target_col, *channel_cols]}
                for c in control_cols:
                    is_flag = set(pd.unique(data[c].dropna())) <= {0, 1}
                    agg[c] = "max" if is_flag else "mean"
                resampler = data.set_index(date_col).resample("W-MON", label="left", closed="left")
                days_per_week = resampler[target_col].count()
                data = resampler.agg(agg)
                # Drop partial weeks at the start / end (fewer than 5 observed days).
                partial = days_per_week < 5
                edge = partial & ((days_per_week.index == days_per_week.index.min())
                                  | (days_per_week.index == days_per_week.index.max()))
                if edge.any():
                    notes.append(f"Dropped {int(edge.sum())} partial week(s) at the start/end of the data.")
                    data = data[~edge.to_numpy()]
                data = data.reset_index()
                notes.append(f"Data looked daily (median gap {median_gap:.0f} day) – resampled to weekly.")
            elif median_gap is not None and median_gap > 8:
                notes.append(f"Median gap between periods is {median_gap:.0f} days (not weekly). "
                             "Adstock decay is then 'per period'; set the seasonal period accordingly.")
    else:
        data = data.reset_index(drop=True)
        data.insert(0, out_date, pd.date_range("2000-01-03", periods=len(data), freq="W-MON"))
        notes.append("No date column – a synthetic weekly index was created from row order. "
                     "If rows are not in time order, adstock estimates are not meaningful.")

    n0 = len(data)
    data = data.dropna(subset=[target_col]).reset_index(drop=True)
    if len(data) < n0:
        notes.append(f"Dropped {n0 - len(data)} rows with a missing target.")
    neg = [c for c in channel_cols if (data[c] < 0).any()]
    if neg:
        notes.append(f"Negative spend clipped to 0 in: {neg}")
        for c in neg:
            data[c] = data[c].clip(lower=0)
    zero = [c for c in channel_cols if float(data[c].sum()) <= 0]
    if zero:
        raise DataLoadError(f"These channels have no spend at all: {zero}. Deselect them.")
    return data, notes


# =========================================================================== #
# Sources                                                                      #
# =========================================================================== #
def load_synthetic(regenerate: bool = False, **kwargs) -> pd.DataFrame:
    """Load the synthetic CSV, generating it on first use."""
    if regenerate or kwargs or not DEFAULT_CSV.exists():
        return save_synthetic_data(DEFAULT_CSV, DEFAULT_TRUTH, **kwargs)
    df = pd.read_csv(DEFAULT_CSV, parse_dates=["date"])
    return df


def read_tabular_file(source: Union[str, Path, io.BytesIO], name: Optional[str] = None) -> pd.DataFrame:
    """Read CSV / TSV / Excel / Parquet / JSON from a path or an uploaded buffer."""
    fname = (name or str(source)).lower()
    try:
        if fname.endswith((".xlsx", ".xls")):
            return pd.read_excel(source)
        if fname.endswith(".parquet"):
            return pd.read_parquet(source)
        if fname.endswith(".json"):
            return pd.read_json(source)
        if fname.endswith(".tsv"):
            return pd.read_csv(source, sep="\t")
        # CSV with delimiter sniffing (handles ; and | separated exports)
        return pd.read_csv(source, sep=None, engine="python")
    except Exception as exc:
        raise DataLoadError(f"Could not read '{name or source}': {exc}") from exc


def list_tabular_files(folder: Union[str, Path]) -> List[Path]:
    """All tabular files under ``folder``, largest first."""
    folder = Path(folder)
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR_EXTENSIONS]
    return sorted(files, key=lambda p: p.stat().st_size, reverse=True)


def load_kaggle_dataset(handle: str = KAGGLE_HANDLE, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Download the Kaggle dataset with ``kagglehub`` and read one file.

    Strategy
    --------
    1. ``kagglehub.dataset_download(handle)`` → local cache folder; pick
       ``file_path`` if given, otherwise the largest tabular file.
    2. Fallback: ``kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, ...)``
       (the adapter used in the dataset's Kaggle snippet).

    Public datasets download anonymously; private ones need Kaggle credentials
    (``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``).

    Returns
    -------
    (DataFrame, str)
        The data and the name of the file that was read.
    """
    try:
        import kagglehub  # imported lazily so the app works without it
    except ImportError as exc:  # pragma: no cover
        raise DataLoadError("kagglehub is not installed – run `pip install -r requirements.txt`.") from exc

    errors = []
    # --- Strategy 1: download the files ------------------------------------
    try:
        folder = Path(kagglehub.dataset_download(handle))
        if file_path:
            target = folder / file_path
            if not target.exists():
                raise DataLoadError(f"'{file_path}' not found in dataset. Available: "
                                    f"{[p.name for p in list_tabular_files(folder)]}")
        else:
            files = list_tabular_files(folder)
            if not files:
                raise DataLoadError(f"No tabular files found in {folder}")
            target = files[0]
        df = read_tabular_file(target, name=target.name)
        logger.info("Loaded Kaggle file %s with shape %s", target, df.shape)
        return df, target.name
    except Exception as exc:
        errors.append(f"dataset_download: {exc}")

    # --- Strategy 2: pandas adapter ----------------------------------------
    try:
        from kagglehub import KaggleDatasetAdapter
        df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, handle, file_path or "")
        return df, file_path or "(default file)"
    except Exception as exc:
        errors.append(f"load_dataset: {exc}")

    raise DataLoadError(
        "Could not load the Kaggle dataset. Check your internet connection or Kaggle credentials.\n"
        + "\n".join(errors)
    )
