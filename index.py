"""Stateless Vercel adapter for HardikMalik14/MMM-budget-engine."""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from data.generator import generate_synthetic_data
from data.loader import coerce_numeric_columns, detect_schema, prepare_dataframe
from models.mmm import ChannelParams, MarketingMixModel, MMMConfig
from models.optimizer import BudgetOptimizer, OptimizerConstraints

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="MMM Budget Engine", version="1.0.0")
log = logging.getLogger(__name__)
# Bound CPU-heavy requests per process; no user data or fitted model is cached.
WORKER = threading.BoundedSemaphore(1)
MAX_BODY = 3_000_000


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source: Literal["synthetic", "upload"] = "synthetic"
    seed: int = Field(42, ge=0, le=10000)
    weeks: int = Field(156, ge=60, le=520)
    file_name: str = Field("", max_length=200)
    file_data: str = Field("", max_length=2_800_000)
    date_col: str | None = Field(None, max_length=200)
    target: str | None = Field(None, max_length=200)
    channels: list[str] | None = Field(None, max_length=8)
    controls: list[str] | None = Field(None, max_length=12)
    auto_tune: bool = True
    trend: bool = True
    seasonality: bool = True
    fourier: int = Field(2, ge=1, le=4)
    alpha: float = Field(0.5, ge=0, le=0.95)
    ec50: float = Field(1.0, ge=0.1, le=4)
    slope: float = Field(1.5, ge=0.5, le=4)
    horizon: int = Field(52, ge=4, le=156)
    budget: float | None = Field(None, gt=0, le=1e12)
    min_share: float = Field(0.05, ge=0, le=1)
    max_share: float = Field(0.5, ge=0, le=1)
    min_mult: float = Field(0.5, ge=0, le=5)
    max_mult: float = Field(1.5, ge=0, le=10)
    auto_relax: bool = False


def records(frame, with_index=False):
    return json.loads((frame.reset_index() if with_index else frame).to_json(orient="records", date_format="iso"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


@app.middleware("http")
async def bound_body(request: Request, call_next):
    if request.method == "POST":
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY:
                return JSONResponse({"detail": "Upload too large. Use a file under 2 MB."}, status_code=413)
        request._body = bytes(body)
    return await call_next(request)


def load_data(o: Options):
    if o.source == "synthetic":
        if o.seed == 42 and o.weeks == 156:
            df = pd.read_csv(ROOT / "data/synthetic_mmm_data.csv")
        else:
            df, _ = generate_synthetic_data(n_weeks=o.weeks, seed=o.seed)
    else:
        try:
            raw = base64.b64decode(o.file_data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid file encoding.") from exc
        if len(raw) > 2_000_000:
            raise ValueError("Use a file under 2 MB.")
        suffix = Path(o.file_name).suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(io.BytesIO(raw), nrows=2001)
        elif suffix == ".xlsx":
            import zipfile
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if sum(f.file_size for f in archive.infolist()) > 20_000_000:
                    raise ValueError("Excel workbook expands beyond the 20 MB limit.")
            df = pd.read_excel(io.BytesIO(raw), nrows=2001)
        else:
            raise ValueError("Upload a CSV or XLSX file.")
    if df.empty or len(df) > 2000 or len(df.columns) > 40:
        raise ValueError("Use 1–2,000 rows and at most 40 columns.")
    if any(not isinstance(c, str) or len(c) > 200 for c in df.columns):
        raise ValueError("Column names must be text under 200 characters.")
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")
    df = coerce_numeric_columns(df)
    numeric = df.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Data contains infinite values. Replace them before uploading.")
    return df


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "public/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "MMM Budget Engine"}


@app.post("/api/schema")
def schema(o: Options):
    try:
        df = load_data(o)
        guess = detect_schema(df)
        return {
            "columns": list(df.columns),
            "numeric": list(df.select_dtypes(include="number").columns),
            "date": guess.date_col, "target": guess.target_col,
            "channels": guess.channel_cols, "controls": guess.control_cols,
            "rows": len(df), "preview": records(df.head(6)), "notes": guess.notes,
        }
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("Unable to read file")
        raise HTTPException(400, "Could not read this file. Check its format and contents.") from exc


def analyze_model(o):
    df = load_data(o)
    guess = detect_schema(df)
    target = o.target or guess.target_col
    date = o.date_col if o.date_col is not None else guess.date_col
    date = None if date == "__row_order__" else date
    channels = o.channels if o.channels is not None else guess.channel_cols
    controls = o.controls if o.controls is not None else guess.control_cols
    if not channels or len(channels) > 8 or len(controls) > 12:
        raise ValueError("Select 1–8 spend channels and at most 12 controls.")
    selected = [target, *channels, *controls]
    if len(selected) != len(set(selected)) or date in selected:
        raise ValueError("Target, date, channels and controls must use separate columns.")
    reserved = {"const", "trend", "fitted", "actual", *(f"season_{s}_{k}" for s in ["sin", "cos"] for k in range(1, 5))}
    if reserved.intersection(channels + controls) or (date is None and "date" in selected):
        raise ValueError("Rename model columns that conflict with internal terms (date, const, trend, season_*, actual, fitted).")
    data, notes = prepare_dataframe(df, date, target, channels, controls)
    if len(data) > 520:
        raise ValueError("After weekly aggregation, use at most 520 periods.")
    date = date or "date"
    cfg = MMMConfig(target_col=target, channel_cols=channels, control_cols=controls,
                    date_col=date, add_trend=o.trend, add_seasonality=o.seasonality, n_fourier=o.fourier)
    params = {ch: ChannelParams(alpha=o.alpha, ec50_ratio=o.ec50, slope=o.slope) for ch in channels}
    model = MarketingMixModel(cfg, params).fit(data, auto_tune=o.auto_tune, n_passes=2)
    optimizer = BudgetOptimizer(model, horizon_weeks=o.horizon)
    constraints = OptimizerConstraints(min_share=o.min_share, max_share=o.max_share,
                                      min_mult=o.min_mult, max_mult=o.max_mult)
    # Keep original constraints; the UI explicitly opts in to relaxation.
    result = optimizer.optimize(o.budget or optimizer.historical_total, constraints,
                                auto_relax=o.auto_relax, n_random_starts=4)
    dec = model.decompose()
    return clean({
        "source": "Synthetic demonstration" if o.source == "synthetic" else o.file_name,
        "metrics": model.fit_metrics(), "target": target, "date": date,
        "channels": channels, "controls": controls, "data": records(model.data_),
        "decomposition": records(dec), "channel_metrics": records(model.channel_metrics(), True),
        "coefficients": records(model.coefficient_table(), True),
        "response_curves": {ch: records(model.response_curve(ch)) for ch in channels},
        "correlation": json.loads(model.data_[[target, *channels, *controls]].corr().to_json()),
        "warnings": list(dict.fromkeys(notes + model.warnings_)),
        "optimization": {"allocation": records(result.allocation, True),
            "total_budget": result.total_budget, "horizon": result.horizon_weeks,
            "current_revenue": result.current_revenue, "optimal_revenue": result.optimal_revenue,
            "lift": result.lift_abs, "lift_pct": result.lift_pct,
            "success": result.success, "message": result.message, "relaxations": result.relaxations},
    })


@app.post("/api/analyze")
def analyze(o: Options):
    if not WORKER.acquire(blocking=False):
        raise HTTPException(429, "Another analysis is running. Try again shortly.")
    try:
        return analyze_model(o)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("Analysis failed")
        raise HTTPException(500, "Analysis failed. Check the deployment logs for details.") from exc
    finally:
        WORKER.release()
