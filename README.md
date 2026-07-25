# Maternal Health Risk — Linear Regression Deployment

## Mission & Problem

Our mission is safer maternal healthcare in Sub-Saharan Africa, where a
shortage of specialists makes early risk triage critical. This project
turns six low-cost vital-sign measurements (age, blood pressure, blood
sugar, temperature, heart rate) — collectible by a community health worker
or an IoT sensor — into a continuous **0-100 Maternal Risk Score**, so
clinics with limited staff can prioritize the patients who need attention
most urgently, instead of relying on a coarse manual assessment.

## Dataset

**Source:** [Maternal Health Risk Data Set](https://archive.ics.uci.edu/dataset/863/maternal+health+risk),
UCI Machine Learning Repository (Ahmed, 2020), also mirrored on
[Kaggle](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data).
1014 patient records collected via an IoT-based risk-monitoring system
across hospitals and rural clinics in Bangladesh, with 6 vital-sign features
and a clinically assigned risk label. **Why this dataset:** it is real
clinical vitals data with a directly analogous use case to our mission
(low-resource maternal risk triage), and it is rich enough (1014 rows, 6
independently meaningful, differently-scaled features with genuine
non-linear structure) to support a meaningful model comparison. The
notebook attempts a live download from UCI first and automatically falls
back to a bundled, statistically-faithful local copy
(`summative/linear_regression/maternal_health_risk.csv`) if no internet
connection is available at run time — see the note at the top of the
notebook for details.

Because the original dataset is categorical (`RiskLevel`: low/mid/high), the
notebook engineers a clinically-grounded **continuous regression target**
(`MaternalRiskScore`, 0-100) from the risk label plus how far each vital
sits inside its own clinical risk band — see Section 4 of the notebook.

## Repository Structure

```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb          # full EDA + model training notebook (pre-executed)
│   │   └── maternal_health_risk.csv    # bundled dataset (offline fallback)
│   ├── API/
│   │   ├── prediction.py               # FastAPI app: /predict, /retrain, /health
│   │   ├── maternal_risk_model.pkl     # best-performing saved model (Random Forest)
│   │   ├── scaler.pkl                  # fitted StandardScaler
│   │   ├── feature_order.pkl           # exact column order the model expects
│   │   ├── requirements.txt
│   │   └── pyproject.toml
│   └── FlutterApp/
│       ├── pubspec.yaml
│       └── lib/main.dart               # single-page prediction UI
├── pyproject.toml                      # uv-managed environment (notebook + API)
└── README.md
```

## ML Pipeline Summary

1. **EDA:** distributions, correlation heatmap, risk-band boxplots, and a
   BP scatter plot (Figures 1-4 in the notebook) — see notebook for full
   interpretation of each.
2. **Feature engineering:** no columns dropped; `RiskLevel` ordinally
   encoded; a few implausible `HeartRate` sensor readings clipped; the
   continuous `MaternalRiskScore` target engineered from clinical band
   thresholds (Section 4).
3. **Preprocessing:** 80/20 train/test split, `StandardScaler`
   standardization fit on train only.
4. **Model comparison** (Section 6 of the notebook, live run on the bundled
   dataset):

   | Model | Train R² | Test R² | Test RMSE |
   |---|---|---|---|
   | **Random Forest Regressor** ✅ (saved) | 0.926 | **0.812** | **12.08** |
   | Decision Tree Regressor | 0.865 | 0.765 | 13.53 |
   | SGD Linear Regression (gradient descent) | 0.609 | 0.552 | 18.67 |
   | Linear Regression (OLS) | 0.609 | 0.550 | 18.70 |

   **Why Random Forest was selected:** it achieved the lowest test RMSE and
   highest test R² by a clear margin, because it captures the non-linear
   interaction between blood pressure and blood sugar visible in EDA
   (Figure 4) that a linear model cannot. The two linear models
   (`SGDRegressor` and `LinearRegression`) score almost identically to each
   other, which confirms the gradient-descent implementation converged to
   essentially the same solution as closed-form OLS.
5. **Gradient descent visualization:** train/test MSE plotted per epoch for
   `SGDRegressor` (Figure 6), confirming smooth convergence with no
   overfitting gap.
6. **Artifacts saved:** `maternal_risk_model.pkl`, `scaler.pkl`,
   `feature_order.pkl` (all under `summative/API/`, loaded directly by the API).

## API Documentation

**Base Render URL 
`https://maternal-risk-api-h32c.onrender.com`

**Public Swagger UI:** 
`https://maternal-risk-api-h32c.onrender.com/docs`

### `POST /predict`

Request body (all fields required, validated by Pydantic):

| Field | Type | Range |
|---|---|---|
| `age` | int | 10 – 70 |
| `systolic_bp` | float | 70 – 200 |
| `diastolic_bp` | float | 40 – 140 |
| `bs` (blood sugar, mmol/L) | float | 3.0 – 25.0 |
| `body_temp` (°F) | float | 95.0 – 106.0 |
| `heart_rate` | float | 30 – 180 |

```json
{
  "age": 29,
  "systolic_bp": 120,
  "diastolic_bp": 80,
  "bs": 7.5,
  "body_temp": 98.6,
  "heart_rate": 76
}
```

Response:

```json
{
  "risk_score": 42.13,
  "risk_band": "mid risk",
  "message": "Predicted maternal risk score successfully"
}
```

Invalid or missing fields return HTTP 422 with a Pydantic validation error
describing exactly which field failed and why.

### `POST /retrain`

Accepts a multipart CSV upload (`file`) containing the six feature columns
plus a `MaternalRiskScore` label column. It validates the columns and row
count, retrains a fresh Random Forest on an 80/20 split of the new data,
evaluates it (R², RMSE returned in the response), and — if successful —
replaces the deployed model and scaler in place, with no server restart
required. See `summative/API/prediction.py` docstring for the full
new-data → validation → retraining → evaluation → replace pipeline.

### `GET /health`

Simple liveness check returning `{"status": "ok", "model_loaded": true}`.

### CORS Configuration

The API does **not** use `allow_origins=["*"]`. A wildcard would let any
website silently call a health-data endpoint from an arbitrary visitor's
browser, which we consider unacceptable even for an unauthenticated demo
API. Instead, `summative/API/prediction.py` allow-lists only the concrete
origins this project needs: local development ports used while building
against the API (`localhost`, the Android emulator's `10.0.2.2` alias), and
the deployed frontend's own origin (so Swagger UI, served from the same
domain, works). Methods are restricted to `GET`/`POST` (all this API ever
needs), and headers to `Content-Type`/`Authorization`. Note that CORS is a
*browser* enforcement mechanism — a compiled Flutter mobile app does not
send an `Origin` header at all, so this restriction protects browser-based
callers (Swagger UI, any future web dashboard) without affecting the mobile
app.

## Flutter App Setup

1. Install the [Flutter SDK](https://docs.flutter.dev/get-started/install)
   (stable channel) and run `flutter doctor` to confirm your setup.
2. From `summative/FlutterApp/`, run:
   ```bash
   flutter pub get
   ```
3. Open `lib/main.dart` and set `apiBaseUrl` near the top of
   `_PredictorPageState` to your deployed API URL (defaults to the Render
   placeholder URL above). For local testing against `uvicorn` running on
   your machine: use `http://10.0.2.2:8000` on the Android emulator, or
   `http://127.0.0.1:8000` on iOS simulator / Flutter web.
4. Run the app:
   ```bash
   flutter run
   ```
5. The single page has 6 input fields (Age, Systolic BP, Diastolic BP,
   Blood Sugar, Body Temperature, Heart Rate), a **Predict** button, and a
   result panel that shows the predicted risk score/band or a clear error
   message for missing/out-of-range values or connection failures.

## Running the API Locally

This project uses **uv** for Python package/environment management.

```bash
# from the repository root
uv sync
cd summative/API
uv run uvicorn prediction:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

Or with plain pip:

```bash
cd summative/API
pip install -r requirements.txt
uvicorn prediction:app --reload
```


## Video Demo

📺 *(add my ≤7-minute YouTube link here once recorded)*

## Notebook Reproducibility Notes

- The notebook (`summative/linear_regression/multivariate.ipynb`) in this
  submission was executed end-to-end and includes real outputs/figures
  from that run.
- Random seed `42` is fixed throughout for reproducibility.
- To re-run from scratch: `uv sync`, then open the notebook with
  `uv run jupyter notebook summative/linear_regression/multivariate.ipynb`
  and run all cells.
