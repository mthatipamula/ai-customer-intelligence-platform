from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from data_pipeline import prepare_data


app = FastAPI(
    title="AI Customer Intelligence Platform",
    version="1.0.0",
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


class TicketRequest(BaseModel):
    ticket_text: str


class ClassificationResponse(BaseModel):
    ticket_text: str
    predicted_category: str
    confidence: float


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health."""

    return {
        "status": "ok"
    }


@app.get("/")
def root():
    """Return API information."""

    return {
        "name": "AI Customer Intelligence Platform",
        "version": "1.0.0",
        "status": "running",
    }


@app.post(
    "/classify",
    response_model=ClassificationResponse,
)
def classify_ticket(request: TicketRequest):
    """Classify a customer support ticket."""

    ticket_text = request.ticket_text.strip().lower()

    if not ticket_text:
        raise ValueError(
            "ticket_text cannot be empty"
        )

    features = vectorizer.transform(
        [ticket_text]
    )

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = float(
        probabilities.max()
    )

    return {
        "ticket_text": ticket_text,
        "predicted_category": prediction,
        "confidence": confidence,
    }


@app.post("/prepare-ticket")
def prepare_ticket(request: TicketRequest):
    """Run basic data preparation on a customer ticket."""

    data_path = PROJECT_ROOT / "data" / "tickets.csv"

    df = prepare_data(str(data_path))

    return {
        "ticket_text": request.ticket_text.strip().lower(),
        "dataset_records": len(df),
        "message": "Ticket prepared successfully",
    }