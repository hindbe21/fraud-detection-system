from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test, threshold=0.3):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1_fraud": f1_score(y_test, y_pred, pos_label=1),
        "recall_fraud": recall_score(y_test, y_pred, pos_label=1),
        "precision_fraud": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics