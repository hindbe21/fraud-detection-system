import pandas as pd
from xgboost import XGBClassifier


def select_top_features(X_train, y_train, feature_names, top_k=20, random_state=42):
    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state
    )
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    selected_features = importance_df.head(top_k)["feature"].tolist()
    return selected_features, importance_df