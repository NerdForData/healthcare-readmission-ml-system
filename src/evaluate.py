import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

try:
    from data_prep import load_data, prepare_features_and_target
except ImportError:
    from src.data_prep import load_data, prepare_features_and_target


def clinical_threshold_analysis(
    data_path: str,
    model_path: str = "models/readmission_pipeline_gb.joblib",
    desired_recall: float = 0.80
):
    # Load model and data
    pipeline = joblib.load(model_path)
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

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, proba)

    # Find threshold achieving desired recall
    idx = np.where(recall >= desired_recall)[0][-1]
    threshold = thresholds[idx]

    return {
        "threshold": float(threshold),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "patients_flagged": int((proba >= threshold).sum()),
        "total_patients": len(y_val)
    }
