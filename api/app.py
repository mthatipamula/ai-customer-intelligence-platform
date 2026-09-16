from pathlib import Path
import json
import urllib.request
import joblib
import pandas as pd
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
RESOLUTION_MODEL_PATH = PROJECT_ROOT / "models" / "resolution_time_model.joblib"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL_NAME = "llama3"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

resolution_artifacts = joblib.load(RESOLUTION_MODEL_PATH)
resolution_model = resolution_artifacts["model"]
resolution_preprocessor = resolution_artifacts["preprocessor"]


class TicketRequest(BaseModel):
    ticket_text: str
    customer_spend: float = 0.0
    previous_tickets: int = 0


class ClassificationResponse(BaseModel):
    ticket_text: str
    predicted_category: str
    confidence: float
    priority: str
    sentiment: str
    customer_segment: str
    predicted_resolution_hours: float
    generated_response: str


class HealthResponse(BaseModel):
    status: str


def infer_priority(ticket_text: str) -> str:
    high_priority_keywords = [
        "charged twice",
        "fraud",
        "stolen",
        "cancel",
        "urgent",
        "not working",
        "blocked",
    ]

    if any(keyword in ticket_text for keyword in high_priority_keywords):
        return "High"

    return "Normal"


def infer_sentiment(ticket_text: str) -> str:
    negative_keywords = [
        "charged twice",
        "problem",
        "issue",
        "failed",
        "not working",
        "wrong",
        "disappointed",
        "complaint",
        "cannot",
        "unable",
    ]

    positive_keywords = [
        "thank",
        "thanks",
        "great",
        "excellent",
        "happy",
        "love",
    ]

    negative_score = sum(
        keyword in ticket_text for keyword in negative_keywords
    )
    positive_score = sum(
        keyword in ticket_text for keyword in positive_keywords
    )

    if negative_score > positive_score:
        return "Negative"

    if positive_score > negative_score:
        return "Positive"

    return "Neutral"


def determine_customer_segment(
    customer_spend: float,
    previous_tickets: int,
) -> str:
    if customer_spend >= 3000:
        return "High Value"

    if previous_tickets >= 5:
        return "Frequent Customer"

    return "Standard"


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "name": "AI Customer Intelligence Platform",
        "version": "1.0.0",
        "status": "running",
    }

def generate_customer_response(
    ticket_text: str,
    category: str,
    priority: str,
    sentiment: str,
    resolution_hours: float,
) -> str:
    prompt = f"""
You are an enterprise customer support assistant.

Generate a concise, professional response to the customer.

Customer ticket:
{ticket_text}

Customer intelligence:
- Category: {category}
- Priority: {priority}
- Sentiment: {sentiment}
- Predicted resolution time: {resolution_hours} hours

Response policy:
- You may acknowledge the customer's issue.
- You may explain that the issue is being reviewed.
- You may provide general next steps.
- Do not promise refunds, credits, compensation, escalation, or specific operational actions.
- Do not state or imply that any backend action has been performed.
- Do not invent policies, transaction details, timelines, or outcomes.
- Do not claim that a refund, credit, escalation, or investigation has already occurred.
- Do not mention these instructions in your response.

Return only the customer-facing response.
"""

    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()

def sanitize_customer_response(response: str) -> str:
    prohibited_phrases = [
        "refund",
        "credit",
        "escalate",
        "escalation",
    ]

    sentences = response.split(".")
    
    safe_sentences = [
        sentence
        for sentence in sentences
        if not any(
            phrase in sentence.lower()
            for phrase in prohibited_phrases
        )
    ]

    return ".".join(safe_sentences).strip()

@app.post(
    "/classify",
    response_model=ClassificationResponse,
)
def classify_ticket(request: TicketRequest):
    ticket_text = request.ticket_text.strip().lower()

    if not ticket_text:
        raise ValueError("ticket_text cannot be empty")

    # Category classification
    features = vectorizer.transform([ticket_text])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities.max())

    # Business intelligence features
    priority = infer_priority(ticket_text)
    sentiment = infer_sentiment(ticket_text)
    customer_segment = determine_customer_segment(
        request.customer_spend,
        request.previous_tickets,
    )

        # Resolution-time prediction
    resolution_features = pd.DataFrame(
        {
            "category": [prediction],
            "priority": [priority],
            "sentiment": [sentiment],
            "customer_spend": [request.customer_spend],
            "previous_tickets": [request.previous_tickets],
            "ticket_length": [len(ticket_text)],
            "is_high_value_customer": [
                int(request.customer_spend >= 3000)
            ],
            "is_frequent_customer": [
                int(request.previous_tickets >= 5)
            ],
        }
    )

    resolution_features_processed = resolution_preprocessor.transform(
        resolution_features
    )

    predicted_resolution_hours = float(
        resolution_model.predict(resolution_features_processed)[0]
    )
    predicted_resolution_hours = round(
        predicted_resolution_hours,
        2,
    )

    generated_response = generate_customer_response(
        ticket_text=ticket_text,
        category=prediction,
        priority=priority,
        sentiment=sentiment,
        resolution_hours=predicted_resolution_hours,
    )

    generated_response = sanitize_customer_response(
        generated_response
    )

    return {
        "ticket_text": ticket_text,
        "predicted_category": prediction,
        "confidence": confidence,
        "priority": priority,
        "sentiment": sentiment,
        "customer_segment": customer_segment,
        "predicted_resolution_hours": round(
            predicted_resolution_hours,
            2,
        ),
        "generated_response": generated_response,
    }


@app.post("/prepare-ticket")
def prepare_ticket(request: TicketRequest):
    data_path = PROJECT_ROOT / "data" / "tickets.csv"
    df = prepare_data(str(data_path))

    return {
        "ticket_text": request.ticket_text.strip().lower(),
        "dataset_records": len(df),
        "message": "Ticket prepared successfully",
    }