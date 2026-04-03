import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
META_PATH = ARTIFACTS_DIR / "preprocessing_meta.json"


class FraudPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler introuvable : {SCALER_PATH}")
        if not META_PATH.exists():
            raise FileNotFoundError(f"Métadonnées introuvables : {META_PATH}")

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.selected_features = self.meta["selected_features"]
        self.full_feature_order = self.meta["full_feature_order"]
        self.amount_mean_train = self.meta["amount_mean_train"]
        self.default_threshold = self.meta["threshold_default"]
        self.model_name = type(self.model).__name__

    def _add_engineered_features_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        data["hour_of_day"] = ((data["Time"] // 3600) % 24).astype(int)
        data["is_night"] = data["hour_of_day"].apply(lambda h: 1 if (h >= 22 or h <= 6) else 0)
        data["amount_log"] = np.log1p(data["Amount"])
        data["amount_to_mean_ratio"] = data["Amount"] / max(self.amount_mean_train, 1e-6)

        return data

    def predict(self, transaction: dict, threshold: float | None = None):
        if threshold is None:
            threshold = self.default_threshold

        df = pd.DataFrame([transaction])
        df = self._add_engineered_features_for_inference(df)

        df = df[self.full_feature_order]

        X_scaled = self.scaler.transform(df)
        scaled_df = pd.DataFrame(X_scaled, columns=self.full_feature_order)
        X_selected = scaled_df[self.selected_features]

        fraud_probability = float(self.model.predict_proba(X_selected)[0, 1])
        is_fraud = fraud_probability >= threshold

        return {
            "fraud_probability": round(fraud_probability, 6),
            "is_fraud": bool(is_fraud),
            "threshold_used": float(threshold),
            "model_name": self.model_name
        }