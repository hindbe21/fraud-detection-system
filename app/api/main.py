from fastapi import FastAPI
from app.api.schemas import TransactionInput, PredictionOutput
from app.api.predictor import FraudPredictor

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude bancaire en temps réel",
    version="1.0.0"
)

predictor = FraudPredictor()


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    payload = transaction.model_dump()
    threshold = payload.pop("threshold", 0.3)
    result = predictor.predict(payload, threshold=threshold)
    return result