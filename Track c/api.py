"""
BankMind Challenge - Track C (System Builder)

FastAPI service wrapping the trained Random Forest model from
train_model.py. Exposes /predict and /health endpoints.

Run with: uvicorn api:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import os

MODEL_PATH = "model.pkl"

model_bundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    if os.path.exists(MODEL_PATH):
        model_bundle = joblib.load(MODEL_PATH)
    else:
        model_bundle = None
    yield


app = FastAPI(
    title="BankMind API",
    description="Predicts whether a bank customer will subscribe to a term deposit.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------
class CustomerFeatures(BaseModel):
    age: int = Field(..., example=42, ge=18, le=110)
    job: str = Field(..., example="technician")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="secondary")
    default: str = Field(..., example="no", description="'yes' or 'no'")
    balance: int = Field(..., example=1500)
    housing: str = Field(..., example="yes", description="'yes' or 'no'")
    loan: str = Field(..., example="no", description="'yes' or 'no'")
    contact: str = Field(..., example="cellular")
    day: int = Field(..., example=15, ge=1, le=31)
    month: str = Field(..., example="may")
    duration: int = Field(..., example=180, ge=0)
    campaign: int = Field(..., example=1, ge=1)
    pdays: int = Field(..., example=-1)
    previous: int = Field(..., example=0, ge=0)
    poutcome: str = Field(..., example="unknown")


class PredictionResponse(BaseModel):
    prediction: str
    probability_yes: float


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Make sure model.pkl exists (run train_model.py first).",
        )

    model = model_bundle["model"]
    encoders = model_bundle["encoders"]
    feature_cols = model_bundle["feature_cols"]

    row = customer.dict()
    df_row = pd.DataFrame([row])

    # Encode categoricals using the same encoders fit during training
    for col, le in encoders.items():
        value = df_row.at[0, col]
        if value not in le.classes_:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown value '{value}' for field '{col}'. "
                       f"Expected one of: {list(le.classes_)}",
            )
        df_row[col] = le.transform([value])

    df_row = df_row[feature_cols]

    pred = model.predict(df_row)[0]
    proba = model.predict_proba(df_row)[0][1]

    return PredictionResponse(
        prediction="yes" if pred == 1 else "no",
        probability_yes=round(float(proba), 4),
    )
