import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame, time_col: str = "Time", amount_col: str = "Amount") -> pd.DataFrame:
    data = df.copy()

    # heure relative dans la fenêtre du dataset
    data["hour_of_day"] = ((data[time_col] // 3600) % 24).astype(int)

    # nuit : 22h -> 6h
    data["is_night"] = data["hour_of_day"].apply(lambda h: 1 if (h >= 22 or h <= 6) else 0)

    # transformation logarithmique du montant
    data["amount_log"] = np.log1p(data[amount_col])

    # ratio du montant par rapport à la moyenne globale
    global_mean = max(data[amount_col].mean(), 1e-6)
    data["amount_to_mean_ratio"] = data[amount_col] / global_mean

    return data