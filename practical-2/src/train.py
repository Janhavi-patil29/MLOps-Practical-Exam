"""
train.py
Baseline training script for the Breast Cancer classifier.
Used to demonstrate a version-controlled ML project structure (MLOps Lab 2).
Dataset: Wisconsin breast cancer CSV (699 samples, 9 features, benign vs malignant)
"""
import joblib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "breast-cancer-wisconsin.csv"
MODEL_PATH = ROOT / "models" / "breast_cancer_model.joblib"
DATA_URL = (
    "https://github.com/jeffheaton/aifh/raw/refs/heads/master/"
    "vol1/python-examples/datasets/breast-cancer-wisconsin.csv"
)


def load_data():
    """Load the Wisconsin breast cancer CSV (downloaded to data/ on first run).

    Columns: id, nine cell-measurement features (1-10), class (2 = benign,
    4 = malignant). bare_nucleoli contains '?' for missing values.
    """
    if not DATA_PATH.exists():
        print(f"{DATA_PATH.name} not found in data/ - downloading ...")
        try:
            raw = pd.read_csv(DATA_URL)
        except Exception as exc:
            raise SystemExit(
                f"Could not download the dataset ({exc}).\n"
                f"Download it manually from:\n  {DATA_URL}\n"
                f"and save it as: {DATA_PATH}"
            )
        DATA_PATH.parent.mkdir(exist_ok=True)
        raw.to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH, na_values="?")
    df = df.drop(columns=["id"])

    if not set(df["class"].unique()) <= {2, 4}:
        raise ValueError("Unexpected values in 'class' column (expected 2 and 4).")
    y = (df["class"] == 4).astype(int)  # 1 = malignant, 0 = benign
    X = df.drop(columns=["class"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, ["benign", "malignant"]


def train_model(X_train, y_train, n_estimators=100, max_depth=None):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        ),
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names=None):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names)
    return acc, report


def main():
    X_train, X_test, y_train, y_test, target_names = load_data()
    model = train_model(X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test, target_names)

    print(f"Accuracy: {acc:.4f}")
    print(report)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
