import argparse
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from data_prep import load_data, prepare_features_and_target, get_preprocessor

def train_model(data_path: str, model_type: str = "logreg"):
    # Load data
    df = load_data(data_path)
    X, y = prepare_features_and_target(df)

    # Split (stratified due to class imbalance)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Preprocessing
    preprocessor = get_preprocessor(X)

    # Choose model
    if model_type == "logreg":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        )
    elif model_type == "gb":
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    else:
        raise ValueError("Unsupported model type")

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)

    print(f"Model: {model_type}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, f"models/readmission_pipeline_{model_type}.joblib")

    return roc_auc, pr_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to diabetic_data.csv"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="logreg",
        choices=["logreg", "gb"]
    )

    args = parser.parse_args()
    train_model(args.data_path, args.model_type)

# We selected Gradient Boosting because it improved both ROC-AUC and precision–recall performance, which is critical under class imbalance in healthcare readmission prediction.

#| Model               | ROC-AUC    | PR-AUC     |
#| ------------------- | ---------- | ---------- |
#| Logistic Regression | 0.6463     | 0.2047     |
#| Gradient Boosting   | 0.6771.    | 0.2306     |
