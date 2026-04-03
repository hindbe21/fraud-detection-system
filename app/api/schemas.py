from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class TransactionInput(BaseModel):
    Time: float = Field(..., description="Temps écoulé depuis la première transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    threshold: Optional[float] = Field(0.3, description="Seuil de décision")


class PredictionOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    model_name: str