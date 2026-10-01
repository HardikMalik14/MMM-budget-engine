# 📈 Marketing Mix Modeling (MMM) & Budget Allocation Engine

An end-to-end, open-source **marketing mix model** with an interactive **budget optimizer**, built for growth-strategy and marketing consultants.

It answers the two questions every CMO asks:

1. **What did each channel actually drive?** – revenue contribution, ROI and marginal ROI per channel, separated from seasonality, trend, holidays and macro conditions.
2. **Where should the next dollar go?** – the revenue-maximising allocation of a total budget, under business constraints, solved in real time.

```
raw weekly spend ─► Geometric adstock ─► Hill saturation ─► OLS (statsmodels) ─► ROI / mROI ─► SciPy SLSQP optimizer
```

---

## ✨ Features

| Area | What you get |
|---|---|
| **Data** | Synthetic 156-week dataset with a *known* ground truth · loader for the Kaggle dataset [`nafees2006/mmm-dataset`](https://www.kaggle.com/datasets/nafees2006/mmm-dataset) via `kagglehub` · upload of any CSV / Excel / Parquet / JSON file · automatic schema detection (date, target, spend, controls) · cleaning of `$1,234`-style strings · daily → weekly resampling |
| **Model** | Geometric adstock (configurable decay α) · Hill saturation (EC50, slope) · OLS with trend + Fourier seasonality + controls · automatic hyper-parameter tuning (coordinate-descent grid search) · p-values, 95 % CIs, VIF, Durbin-Watson · hold-out (out-of-time) validation |
| **Diagnostics** | Actual vs fitted, residuals, revenue decomposition, ROI vs marginal ROI, response curves, adstock decay curves, full statsmodels summary, ground-truth recovery check |
| **Optimizer** | `scipy.optimize.minimize` (SLSQP, multi-start) · share-of-budget bounds (default 5 %–50 %) · baseline-relative bounds (default ±50 %) · per-channel overrides · transparent auto-relaxation of infeasible constraints · budget frontier |
| **Engineering** | Modular, typed, documented code · Streamlit caching · error handling surfaced in the UI · 20 unit tests (`pytest`) |

---

## 🧠 Methodology

### 1. Adstock – advertising carry-over

Advertising does not stop working the week it airs. **Geometric adstock** models this memory:

$$A_t = x_t + \alpha \, A_{t-1}, \qquad 0 \le \alpha < 1$$

* `x_t` – spend in week *t*; `α` – retention (decay) rate.
* `α = 0` → no carry-over (typical of search); `α = 0.7–0.8` → long memory (typical of TV).
* **Half-life** = `ln(0.5) / ln(α)` weeks, i.e. the time for an impulse to lose half its effect.
* By default the output is **normalised** by `(1 − α)` so the weights sum to 1: a constant weekly spend `s` converges to an adstock of exactly `s`. This keeps the transformed series in "effective weekly dollars" and makes saturation parameters comparable across decay rates.

Implementation: `models.mmm.geometric_adstock` (an exact IIR filter via `scipy.signal.lfilter`).

### 2. Hill function – diminishing returns (saturation)

Doubling spend rarely doubles revenue. The **Hill function** maps adstocked spend to a response between 0 and 1:

$$S(A) = \frac{A^{s}}{A^{s} + \text{EC50}^{s}}$$

* **EC50** – the adstocked spend at which the channel reaches *50 % of its maximum effect* (half-saturation point). In the app it is set as a ratio of the channel's mean adstocked spend (`ec50_ratio = 1` → "half-saturated at typical spend") and frozen in dollars after fitting.
* **slope `s`** – curve shape: `s ≤ 1` is concave (every extra dollar is worth less than the last); `s > 1` is S-shaped (a threshold of pressure is needed before the channel "wakes up").

Implementation: `models.mmm.hill_saturation`.

### 3. Regression

$$\text{revenue}_t = \beta_0 + \sum_{c} \beta_c\, S_c\big(A_c(x_{c,t})\big) + \sum_k \gamma_k\, z_{k,t} + \delta\,\text{trend}_t + \sum_{j=1}^{J}\big[a_j \sin(2\pi j t/52.18) + b_j \cos(2\pi j t/52.18)\big] + \varepsilon_t$$

* Estimated by **ordinary least squares** with `statsmodels.OLS` → coefficients, standard errors, t-stats, p-values, confidence intervals, R², adjusted R², F-test, AIC/BIC.
* Because `S ∈ [0, 1)`, a media coefficient `β_c` reads as **the maximum weekly revenue the channel can drive at full saturation**.
* Continuous controls `z` (e.g. a macro index) are **centred** on their mean so their contribution is the effect of *deviating* from normal conditions and the intercept keeps the baseline level; 0/1 flags (holidays) are kept as-is.
* Trend and Fourier seasonality absorb organic growth and yearly cycles so they are not wrongly credited to media.

### 4. Hyper-parameter tuning

α, EC50 and slope enter non-linearly, so they cannot be estimated by OLS directly. With **Auto-tune** enabled, the engine runs a **coordinate-descent grid search**: for one channel at a time it evaluates every (α, EC50, slope) combination on a grid while holding the other channels fixed, keeping the combination with the highest adjusted R². Candidates that would give a channel a *negative* effect are rejected (they are rarely plausible and break optimisation). Two passes over all channels are run; ~3,000 fast `numpy.linalg.lstsq` fits take well under a second. You can switch auto-tune off and set every parameter by hand in the sidebar.

### 5. Attribution & ROI

For each channel *c*:

| Metric | Definition | How to read it |
|---|---|---|
| Contribution | `Σ_t β_c · S_c,t` | Revenue attributed to the channel |
| Contribution % | contribution ÷ total fitted revenue (and ÷ total media revenue) | Share of the business the channel explains |
| **ROI** | contribution ÷ spend | Average revenue per $1 – "did the channel pay back?" |
| **Marginal ROI** | Δ contribution ÷ Δ spend for a +1 % spend change | Revenue from the **next** $1 – "should we spend more?" |

Average ROI looks backwards; **marginal ROI drives reallocation**. A channel can have a great ROI and still be saturated (low mROI).

### 6. Budget optimisation

Given a total budget `B` for a planning horizon of `H` weeks:

$$\max_{b}\; \sum_c \sum_{t \in H} \beta_c\, S_c\big(A_c(x_{c,t}(b_c))\big) \quad \text{s.t.} \quad \sum_c b_c = B, \qquad lo_c \le b_c \le hi_c$$

* **Flighting is preserved**: each channel's budget is spread over the horizon using its *historical weekly pattern* (TV bursts stay bursts), and spend before the horizon is kept so carry-over into the horizon is counted.
* Non-media revenue (base, trend, seasonality, controls) is unaffected by allocation and added as a constant.
* Solved with `scipy.optimize.minimize(method="SLSQP")` on budget **shares** (well-scaled), from several starting points (historical mix, equal split, random Dirichlet draws) because S-shaped Hill curves make the problem non-concave; the best solution is polished by an exact projection onto the constraint set and is never worse than the current mix.
* At the optimum, channels that are not at a bound have **equal marginal ROI** – the textbook optimality condition, displayed in the app.

**Constraints** (the tighter bound wins):

| Family | Default | Meaning |
|---|---|---|
| Share of total budget | 5 % – 50 % | No channel gets less than 5 % or more than 50 % of the total budget |
| Change vs. baseline | 0.5× – 1.5× | No channel moves more than ±50 % from its baseline |
| Per-channel overrides | – | Lock a channel (1.0×–1.0×) or give it more room |

The *baseline* is, by default, the historical mix applied to the chosen budget ("business-as-usual at this budget"); toggle it off to anchor bounds to historical dollars instead. If the bounds cannot all be met (e.g. upper bounds add up to less than the budget), the optimizer **relaxes them step by step and tells you exactly what it relaxed**, or raises a clear error if auto-relax is off.

---

## 📦 Data

### Synthetic demo data (default)

`data/generator.py` simulates **156 weeks** (Jan 2023 – Dec 2025) of:

| Column | Behaviour |
|---|---|
| `date`, `week` | Monday week starts and a 1…156 index |
| `paid_search_spend` | Always-on, mildly seasonal, with occasional budget-cap weeks |
| `paid_social_spend` | Growing always-on spend with quarterly campaign pulses and creative-refresh pauses |
| `influencer_spend` | Sporadic 2–4 week campaigns, near zero otherwise |
| `tv_spend` | Flighted on/off bursts with a heavy Q4 flight every year |
| `macro_economic_index` | Mean-reverting index around 100 |
| `holiday_flag` | 1 for weeks containing New Year, Memorial Day, July 4th, Labor Day, Thanksgiving / Black Friday / Cyber Monday, Christmas |
| `total_revenue` | Base + trend + yearly seasonality + holiday lift + macro effect + **adstocked & saturated media effects** + AR(1) noise |

The true parameters (α, EC50, slope, β, true ROI, true contribution) are written to `data/synthetic_ground_truth.json`, and the dashboard compares them with the estimates. With the default seed the auto-tuned model reaches **R² ≈ 0.97, hold-out MAPE ≈ 1.6 %**, and recovers:

| Channel | True ROI | Estimated ROI |
|---|---|---|
| Paid Search | 1.63 | 1.23 |
| Paid Social | 1.01 | 0.99 |
| Influencer | 1.38 | 1.38 |
| TV | 1.08 | 0.90 |

Recovery is good but not perfect – exactly as in real life. Always-on channels are partly confounded with the baseline; that is why the generator includes weeks with unusually low spend (budget caps, pauses), and why real MMM programmes benefit from deliberate spend variation and incrementality tests.

Regenerate or customise:

```bash
python data/generator.py                        # default 156 weeks, seed 42
python data/generator.py --weeks 208 --seed 7   # 4 years, different seed
```

### Kaggle dataset

Select **Kaggle: nafees2006/mmm-dataset** in the sidebar. `data/loader.py` downloads it with `kagglehub` (public datasets need no credentials; for private ones set `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`), picks the largest tabular file (or the file you name), and auto-detects the column roles. You can review and change the mapping under **Column mapping**. If the download fails (offline, firewall), the app says so and falls back to the synthetic data.

Equivalent standalone code:

```python
from data.loader import load_kaggle_dataset, detect_schema
df, filename = load_kaggle_dataset("nafees2006/mmm-dataset")
print(filename, detect_schema(df))
```

### Your own data

Upload a CSV / Excel file with one row per period:

* a **date** column (daily data is resampled to weekly automatically; without a date, row order is used),
* a **target** column (revenue, sales, conversions…),
* one **spend** column per channel,
* optional **controls** (price, promotions, holidays, macro indices, competitor activity…).

Currency strings such as `$1,234.50`, multiple rows per date (e.g. per region) and partial first/last weeks are handled for you.

---

## 🗂️ Project structure

```
mmm-budget-engine/
├── app.py                      # Streamlit dashboard (3 tabs)
├── data/
│   ├── __init__.py
│   ├── generator.py            # synthetic data with known ground truth (CLI + API)
│   ├── loader.py               # Kaggle / upload loaders, schema detection, cleaning
│   ├── synthetic_mmm_data.csv  # generated demo data (156 weeks)
│   └── synthetic_ground_truth.json
├── models/
│   ├── __init__.py
│   ├── mmm.py                  # adstock, Hill, OLS model, diagnostics, ROI
│   └── optimizer.py            # SciPy constrained budget optimizer
├── tests/
│   └── test_engine.py          # 20 unit tests
├── .streamlit/config.toml      # theme & server settings
├── requirements.txt
├── pytest.ini
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Installation & running

Requires **Python 3.10 – 3.13**.

```bash
# 1. Get the code
git clone https://github.com/<your-username>/mmm-budget-engine.git
cd mmm-budget-engine

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) regenerate the synthetic data
python data/generator.py

# 5. Launch the dashboard
streamlit run app.py               # opens http://localhost:8501

# 6. Run the tests
pytest
```

### Deploying

* **Streamlit Community Cloud** – push to GitHub, create a new app pointing at `app.py`. No secrets are needed for the synthetic or public Kaggle data.
* **Docker / any server** – `pip install -r requirements.txt && streamlit run app.py --server.port 8501 --server.address 0.0.0.0`.

---

## 🖥️ Using the dashboard

**Sidebar** – choose the data source, check the column mapping, toggle auto-tune / trend / seasonality, or set α, EC50 and slope per channel by hand.

**Tab 1 · Data Exploration & Overview** – headline KPIs, revenue and stacked spend over time (holiday weeks marked), spend trend small-multiples with an 8-week average, correlation heatmap, spend mix, spend-vs-revenue scatter, descriptive statistics and the prepared data (downloadable).

**Tab 2 · Model Fit & ROI Diagnostics** – R², adjusted R², in-sample and hold-out MAPE, Durbin-Watson, F-test; actual vs fitted with residuals; channel table (spend, contribution, contribution %, ROI, marginal ROI, p-value, α, half-life, EC50, slope); ROI vs marginal ROI; revenue decomposition; media contribution over time; response curves; adstock decay curves; coefficient table with significance and VIF; full statsmodels summary; tuning log; ground-truth check.

**Tab 3 · Budget Allocation Simulator** – pick the planning horizon and total budget, set share and baseline limits (plus per-channel overrides); the optimizer re-runs instantly and shows current-mix vs optimal revenue, media revenue lift and ROI, spend and contribution comparisons with allowed ranges, each channel's position on its response curve, a downloadable allocation plan, and an optional budget frontier.

---

## 🔌 Using the engine from Python

```python
import pandas as pd
from models.mmm import MarketingMixModel, MMMConfig
from models.optimizer import BudgetOptimizer, OptimizerConstraints

df = pd.read_csv("data/synthetic_mmm_data.csv", parse_dates=["date"])
cfg = MMMConfig(
    target_col="total_revenue",
    channel_cols=["paid_search_spend", "paid_social_spend", "influencer_spend", "tv_spend"],
    control_cols=["macro_economic_index", "holiday_flag"],
    date_col="date",
)
mmm = MarketingMixModel(cfg).fit(df, auto_tune=True)
print(mmm.fit_metrics())
print(mmm.channel_metrics()[["roi", "marginal_roi", "contribution_pct_revenue"]])

opt = BudgetOptimizer(mmm, horizon_weeks=52)
result = opt.optimize(opt.historical_total * 1.1,
                      OptimizerConstraints(min_share=0.05, max_share=0.50, min_mult=0.5, max_mult=1.5))
print(f"Revenue lift: {result.lift_pct:+.2%}")
print(result.allocation[["current_spend", "optimal_spend", "optimal_marginal_roi"]])
```

---

## ⚠️ Assumptions & limitations

* **Correlation ≠ causation.** MMM is observational. Highly correlated channels, always-on spend with little variation, or missing drivers can bias estimates. Validate big decisions with geo-lift or holdout experiments.
* **OLS gives point estimates.** Hyper-parameters are chosen by grid search, not with full uncertainty. For Bayesian MMM with priors and credible intervals see Google's Meridian or PyMC-Marketing.
* **Response curves are held fixed** in the optimizer; creative, pricing and competitive changes can shift them.
* **Optimisation is within the observed range.** Results far outside historical spend levels are extrapolations; the baseline bounds exist for that reason.
* Carry-over generated *after* the planning horizon is not counted (slightly understating long-memory channels such as TV).
* Hold-out validation keeps the tuned hyper-parameters and EC50 scale from the full fit (a small, documented look-ahead).

---

## 🛠️ Tech stack

Python · Streamlit · pandas · NumPy · SciPy (`signal.lfilter`, `optimize.minimize` SLSQP) · statsmodels (OLS, Durbin-Watson, VIF) · scikit-learn (metrics) · matplotlib · seaborn · kagglehub · pytest

## 📄 License

MIT – free to use, modify and distribute.
