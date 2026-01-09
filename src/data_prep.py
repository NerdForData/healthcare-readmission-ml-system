import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Replace missing markers
    df = df.replace("?", np.nan)

    # Define binary target: readmitted within 30 days
    df["readmitted_30"] = df["readmitted"].apply(
        lambda x: 1 if x == "<30" else 0
    )

    return df


def prepare_features_and_target(df: pd.DataFrame):
    y = df["readmitted_30"]

    # Drop identifiers and leakage-prone columns
    X = df.drop(
        columns=[
            "readmitted",
            "readmitted_30",
            "encounter_id",
            "patient_nbr"
        ],
        errors="ignore"
    )

    return X, y


def get_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor
