from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURE_COLUMNS = ["M63391", "T62947", "D14812", "T51250", "H66976", "X55362"]
LABEL_COLUMN = "label"


def find_project_file(relative_path: str) -> Path:
    """Find a project file both locally and inside Docker."""
    script_dir = Path(__file__).resolve().parent
    candidate_roots = [Path.cwd(), script_dir, script_dir.parent]

    for root in candidate_roots:
        candidate = root / relative_path
        if candidate.exists():
            return candidate

    return Path.cwd() / relative_path


def find_models_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidate_roots = [Path.cwd(), script_dir, script_dir.parent]

    for root in candidate_roots:
        if (root / "data").exists() or (root / "models").exists():
            return root / "models"

    return Path.cwd() / "models"


def main() -> None:
    data_path = find_project_file("data/colon_cancer.csv")
    models_dir = find_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Colon Cancer Precision Oncology - Training")
    print(f"Dataset path: {data_path}")
    print(f"Models directory: {models_dir}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("Required gene columns found:")
    for gene in FEATURE_COLUMNS:
        print(f"  - {gene}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    matrix_df = pd.DataFrame(matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    print("\nEvaluation results")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion matrix:")
    print(matrix_df)
    print("\nClassification report:")
    print(report)

    joblib.dump(model, models_dir / "model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")

    print("Saved artifacts:")
    print(f"  - {models_dir / 'model.pkl'}")
    print(f"  - {models_dir / 'scaler.pkl'}")
    print(f"  - {models_dir / 'label_encoder.pkl'}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
