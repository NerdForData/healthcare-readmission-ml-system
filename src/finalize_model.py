import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from data_prep import load_data, prepare_features_and_target
except ImportError:
    from src.data_prep import load_data, prepare_features_and_target


def finalize_clinical_model(
    data_path: str,
    model_path: str = "models/readmission_pipeline_gb.joblib",
    desired_recall: float = 0.80
):
    # Load trained model
    pipeline = joblib.load(model_path)

    # Load data
    df = load_data(data_path)
    X, y = prepare_features_and_target(df)

    # Validation split
    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Predict probabilities
    proba = pipeline.predict_proba(X_val)[:, 1]

    # Compute threshold for high recall
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_val, proba)

    idx = (recall >= desired_recall).nonzero()[0][-1]
    threshold = float(thresholds[idx])

    # Save final artifacts
    joblib.dump(pipeline, "models/final_readmission_pipeline.joblib")

    with open("models/clinical_threshold.json", "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "desired_recall": desired_recall
            },
            f,
            indent=4
        )

    print("Final clinical model and threshold saved.")
    print(f"Threshold: {threshold:.4f}")
    print(f"Recall target: {desired_recall}")


if __name__ == "__main__":
    finalize_clinical_model(
        data_path="data/raw/diabetic_data.csv"
    )
