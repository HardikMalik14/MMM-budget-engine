# Deploy the ORIGINAL MMM-budget-engine repository

Repository: https://github.com/HardikMalik14/MMM-budget-engine

The update adds a Vercel entrypoint (`index.py`) while retaining the original Streamlit `app.py` and Python modeling code. `pyproject.toml` explicitly selects `index:app` so Vercel does not launch the Streamlit file.

## Deploy

1. Commit this update to the original repository and push to `main`.
2. At https://vercel.com/new, import **HardikMalik14/MMM-budget-engine**.
3. Framework: **FastAPI**. Root directory: `./`. Leave Build Command, Install Command, and Output Directory overrides disabled. No environment variables are required.
4. Click Deploy; when ready, click Visit, then Run analysis.
5. Check the Overview, Model diagnostics, and Budget allocation views. The default demonstration has 156 weeks and four channels.
6. Add the generated `.vercel.app` address to your portfolio and the GitHub repository's About → Website field.

Reference: https://vercel.com/docs/frameworks/backend/fastapi

## Local commands

For the Vercel web app: `pip install -r requirements.txt`, then `uvicorn index:app --reload`.

For the original Streamlit app: `pip install -r requirements-streamlit.txt`, then `streamlit run app.py`.

For tests: `pip install -r requirements-dev.txt`, then `python -m pytest -q`.

## What's included

- Synthetic weekly data (default 156 weeks), with configurable seed and length.
- CSV and XLSX uploads, column mapping, automatic numeric conversion, and weekly aggregation.
- Original adstock/Hill/OLS fitting, including the original two-pass tuning grid.
- Manual settings shared across channels, trend and Fourier seasonality.
- Observed vs fitted revenue, spend charts, channel contribution/ROI, response curves, coefficients, correlations, and engine warnings.
- Budget and planning horizon controls; share and baseline-relative constraints; the original multi-start SLSQP optimizer.
- Optional, explicit automatic relaxation of infeasible constraints.
- JSON result download.
- A responsive interface with no external JavaScript or CSS dependencies.

## Scope and limits

The interface is a portfolio edition rather than a complete reproduction of every Streamlit widget. It supports CSV/XLSX instead of all upstream upload formats, and manual parameters apply to every channel. It does not include a live Kaggle download button, per-channel manual parameter overrides, holdout-validation UI, or the budget-sweep UI. To use Kaggle data, download a CSV yourself and upload it. The modeling core remains unchanged.

Uploads are capped at 2 MB, 2,000 raw rows, 40 columns and 520 aggregated weekly periods. Models allow up to eight spend channels and twelve controls. Data is sent to your deployment for computation; this app does not persist uploads or results to a database. The browser keeps the current analysis only until refresh. A process handles one analysis at a time and returns a retry message when busy.

The monetary display uses dollars to match the included synthetic dataset. Upload values must use consistent currency units. Fit metrics are in-sample, attributed ROI is revenue ROI rather than profit ROI, and the allocation scenario reuses recent historical weekly patterns; it is not a forecast or causal experiment. Underlying statistical caveats from the original project still apply.

## If deployment fails

Open the deployment in Vercel and expand **Build Logs**. Share the first error and the lines around it for diagnosis.

- **404:** confirm that `app.py` is at the repository root and that the root directory is `./`.
- **ModuleNotFoundError:** check that `data` and `models`, including their `__init__.py` files, were uploaded; check `requirements.txt` too.
- **Function timeout:** disable auto-tuning and try the synthetic default dataset again.
- **Infeasible constraints:** reduce the minimum shares / increase maximum shares, widen baseline bounds, or explicitly enable relaxation. For one channel, its maximum share must allow 100%.
- **File upload error:** use CSV or XLSX within the published limits and review the column mapping.

For a portfolio README, add the deployed URL to the repository's **About → Website** field, and put a `Live demo` link near the top of the README. Use the GitHub repository URL separately as the source-code link.


## Validation

26 Python tests passed, including unchanged upstream engine tests and adapter checks. The same browser UI was checked for analysis, navigation, recalculation, export, and mobile width. The public Vercel build must still be verified in your account.
