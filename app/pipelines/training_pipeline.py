import json
from pathlib import Path
import joblib
import mlflow

from app.core.config import load_config
from app.core.logger import get_logger
from app.data.load_data import load_dataset
from app.data.feature_engineering import add_engineered_features
from app.data.preprocess import split_data, scale_data, apply_smote
from app.data.feature_selection import select_top_features
from app.models.registry import get_models
from app.models.evaluate import evaluate_model

logger = get_logger(__name__)


def run_training_pipeline():
    config = load_config()

    df = load_dataset(config["data"]["raw_path"])
    logger.info(f"Dataset chargé: {df.shape}")

    df = add_engineered_features(
        df,
        time_col=config["features"]["time_column"],
        amount_col=config["features"]["amount_column"]
    )

    target = config["features"]["target"]

    X_train, X_test, y_train, y_test = split_data(
        df,
        target=target,
        test_size=config["project"]["test_size"],
        random_state=config["project"]["random_state"]
    )

    feature_names = X_train.columns.tolist()

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    selected_features, importance_df = select_top_features(
        X_train_scaled,
        y_train,
        feature_names=feature_names,
        top_k=config["features"]["selected_top_k"],
        random_state=config["project"]["random_state"]
    )

    selected_idx = [feature_names.index(f) for f in selected_features]
    X_train_selected = X_train_scaled[:, selected_idx]
    X_test_selected = X_test_scaled[:, selected_idx]

    if config["sampling"]["use_smote"]:
        X_train_final, y_train_final = apply_smote(
            X_train_selected,
            y_train,
            sampling_strategy=config["sampling"]["smote_sampling_strategy"],
            random_state=config["project"]["random_state"]
        )
    else:
        X_train_final, y_train_final = X_train_selected, y_train

    mlflow.set_tracking_uri(
        config["MLFLOW_TRACKING_URI"] if "MLFLOW_TRACKING_URI" in config else "http://127.0.0.1:5000"
    )
    mlflow.set_experiment(config["mlflow"].get("experiment_name", "fraud_detection_experiment"))

    best_model = None
    best_name = None
    best_metrics = None
    best_score = -1

    models = get_models(random_state=config["project"]["random_state"])

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Entraînement : {model_name}")
            model.fit(X_train_final, y_train_final)

            metrics = evaluate_model(
                model,
                X_test_selected,
                y_test,
                threshold=config["training"]["threshold_default"]
            )

            mlflow.log_params({
                "model_name": model_name,
                "top_k_features": config["features"]["selected_top_k"],
                "threshold": config["training"]["threshold_default"]
            })

            mlflow.log_metrics({
                "roc_auc": float(metrics["roc_auc"]),
                "f1_fraud": float(metrics["f1_fraud"]),
                "recall_fraud": float(metrics["recall_fraud"]),
                "precision_fraud": float(metrics["precision_fraud"])
            })

            cm_path = Path("artifacts/confusion_matrix.json")
            cm_path.parent.mkdir(parents=True, exist_ok=True)
            cm_path.write_text(json.dumps(metrics["confusion_matrix"]), encoding="utf-8")
            mlflow.log_artifact(str(cm_path))

            score = 0.7 * metrics["recall_fraud"] + 0.3 * metrics["f1_fraud"]
            if score > best_score:
                  best_score = score
                  best_model = model
                  best_name = model_name
                  best_metrics = metrics

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(best_model, artifacts_dir / "best_model.joblib")
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")

    (artifacts_dir / "selected_features.json").write_text(
        json.dumps(selected_features, indent=2),
        encoding="utf-8"
    )

    importance_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    preprocessing_meta = {
        "amount_mean_train": float(X_train["Amount"].mean()),
        "full_feature_order": feature_names,
        "selected_features": selected_features,
        "threshold_default": float(config["training"]["threshold_default"]),
        "best_model_name": best_name
    }

    (artifacts_dir / "preprocessing_meta.json").write_text(
        json.dumps(preprocessing_meta, indent=2),
        encoding="utf-8"
    )

    logger.info(f"Meilleur modèle: {best_name}")
    logger.info(f"Métriques: {best_metrics}")

    return {
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "selected_features": selected_features
    }


if __name__ == "__main__":
    run_training_pipeline()