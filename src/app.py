"""
FastAPI service for Healthcare Readmission Risk Prediction.

⚠️ IMPORTANT:
This service provides clinical decision support only.
It is NOT a diagnostic or treatment system.

Artifacts loaded:
- models/final_readmission_pipeline.joblib
- models/clinical_threshold.json

Run:
    PYTHONPATH=$(pwd) uvicorn src.app:app --reload

Docs:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
from typing import List, Optional


# -----------------------
# Load model & threshold
# -----------------------

MODEL_PATH = "models/final_readmission_pipeline.joblib"
THRESHOLD_PATH = "models/clinical_threshold.json"

pipeline = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold_config = json.load(f)

THRESHOLD = threshold_config["threshold"]
DESIRED_RECALL = threshold_config["desired_recall"]


# -----------------------
# FastAPI app
# -----------------------

app = FastAPI(
    title="Hospital Readmission Risk API",
    description=(
        "Predict 30-day hospital readmission risk for clinical decision support. "
        "This system is intended to assist care teams and should not be used "
        "as a standalone medical decision tool."
    ),
    version="1.0"
)


# -----------------------
# Request schemas
# -----------------------

class Patient(BaseModel):
    race: str
    gender: str
    age: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    payer_code: Optional[str] = None
    medical_specialty: Optional[str] = None
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: str
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None
    number_diagnoses: int


class PatientBatch(BaseModel):
    patients: List[Patient]


# -----------------------
# Endpoints
# -----------------------

@app.get("/")
def health_check():
    return {
        "status": "running",
        "model": "GradientBoosting",
        "threshold": THRESHOLD,
        "recall_target": DESIRED_RECALL,
        "warning": "Decision support only — not for autonomous medical decisions"
    }


@app.post("/predict")
def predict_readmission(patient: Patient):
    df = pd.DataFrame([patient.dict()])

    risk_score = pipeline.predict_proba(df)[0][1]
    high_risk_flag = int(risk_score >= THRESHOLD)

    return {
        "readmission_risk_score": float(risk_score),
        "high_risk_flag": high_risk_flag,
        "threshold_used": THRESHOLD,
        "clinical_recommendation": (
            "High risk: consider post-discharge follow-up"
            if high_risk_flag == 1
            else "Standard discharge workflow"
        ),
        "disclaimer": "This output is intended for clinical decision support only."
    }


@app.post("/predict-batch")
def predict_readmission_batch(batch: PatientBatch):
    df = pd.DataFrame([p.dict() for p in batch.patients])

    risk_scores = pipeline.predict_proba(df)[:, 1]
    flags = (risk_scores >= THRESHOLD).astype(int)

    results = []
    for i in range(len(df)):
        results.append({
            "patient_index": i,
            "readmission_risk_score": float(risk_scores[i]),
            "high_risk_flag": int(flags[i]),
            "clinical_recommendation": (
                "High risk: consider post-discharge follow-up"
                if flags[i] == 1
                else "Standard discharge workflow"
            )
        })

    return {
        "patients_scored": len(results),
        "threshold_used": THRESHOLD,
        "recall_target": DESIRED_RECALL,
        "results": results,
        "disclaimer": "This output is intended for clinical decision support only."
    }
