from pathlib import Path
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    return pd.read_csv(file_path)