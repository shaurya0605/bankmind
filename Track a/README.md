# BankMind Challenge - Track A, B & C

EDA, an interactive Streamlit dashboard, an ML model comparing
Logistic Regression vs Random Forest, and a FastAPI service exposing
that model, all built on the UCI Bank Marketing Dataset for the VITB
AI Innovators Hub screening task.

## What's in here

**Track A - Data Analyst**
- `eda.py` – loads the data, checks shape/dtypes/missing values/class
  distribution, and saves the four required plots into `plots/`.
- `app.py` – interactive Streamlit dashboard with filters (job, age
  range, housing loan) showing the same four insights live.

**Track B - ML Engineer**
- `train_model.py` – focused EDA, preprocessing, trains and evaluates
  Logistic Regression (baseline) and Random Forest (main model),
  prints classification reports, feature importances, and 5 sample
  predictions, then saves the trained model to `model.pkl`.

**Track C - System Builder**
- `api.py` – FastAPI service that loads `model.pkl` and exposes
  `/health` and `/predict` endpoints. Run `train_model.py` first to
  produce `model.pkl`.

**Shared**
- `EXPLANATION.md` – answers to the reflection questions.
- `requirements.txt` – Python dependencies.

## Setup

1. Download `bank-full.csv` from the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
   and place it in this folder (same folder as `app.py`).

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the EDA script

```bash
python eda.py
```

This prints the data checks to the console and saves four plots into
the `plots/` folder.

## Running the dashboard

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).
Use the sidebar to filter by job, age range, and housing loan status —
the metrics and all four charts update live.

## Running the ML model (Track B)

```bash
python train_model.py
```

This prints the data checks, trains both models, prints accuracy/
precision/recall/F1 and a full classification report + confusion
matrix for each, prints Random Forest feature importances, prints 5
sample predictions with probability scores, and saves the trained
Random Forest model to `model.pkl`.

## Running the API (Track C)

Make sure `model.pkl` exists first (run `train_model.py` above), then:

```bash
uvicorn api:app --reload
```

The API will be live at `http://127.0.0.1:8000`. Interactive docs
(Swagger UI) are available at `http://127.0.0.1:8000/docs`.

### Endpoints

**`GET /health`** — checks the service and model are up.

```bash
curl http://127.0.0.1:8000/health
```

Response:
```json
{"status": "ok", "model_loaded": true}
```

**`POST /predict`** — predicts whether a customer will subscribe.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 180,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }'
```

Response:
```json
{"prediction": "no", "probability_yes": 0.1632}
```

If a field is missing, the API returns a `422` with details on which
field is missing. If a categorical field (e.g. `job`) has a value the
model wasn't trained on, the API returns a `400` listing the valid
options.

## Notes

- The dataset uses `;` as the column separator (not a comma) — already
  handled in both scripts via `sep=";"`.
- `bank-full.csv` is not included in this repo (per dataset license /
  size); download it separately as described above.
