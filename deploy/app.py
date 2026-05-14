from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


FEATURE_COLUMNS = ["M63391", "T62947", "D14812", "T51250", "H66976", "X55362"]
MODEL_NOT_FOUND_MESSAGE = "Model not found. Please run the training service first."

app = Flask(__name__)

ARTIFACTS = {
    "model": None,
    "scaler": None,
    "label_encoder": None,
}


def find_models_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidate_roots = [Path.cwd(), script_dir, script_dir.parent]

    for root in candidate_roots:
        if (root / "models").exists():
            return root / "models"

    return Path.cwd() / "models"


MODELS_DIR = find_models_dir()


def load_artifacts() -> tuple[bool, str | None]:
    model_path = MODELS_DIR / "model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    encoder_path = MODELS_DIR / "label_encoder.pkl"

    required_paths = [model_path, scaler_path, encoder_path]
    if not all(path.exists() for path in required_paths):
        return False, MODEL_NOT_FOUND_MESSAGE

    try:
        if ARTIFACTS["model"] is None:
            ARTIFACTS["model"] = joblib.load(model_path)
            ARTIFACTS["scaler"] = joblib.load(scaler_path)
            ARTIFACTS["label_encoder"] = joblib.load(encoder_path)
    except Exception as exc:
        ARTIFACTS["model"] = None
        ARTIFACTS["scaler"] = None
        ARTIFACTS["label_encoder"] = None
        return False, f"Unable to load model artifacts: {exc}"

    return True, None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    result_type = None
    error = None
    values = {gene: "" for gene in FEATURE_COLUMNS}

    model_ready, model_error = load_artifacts()
    if model_error:
        error = model_error

    if request.method == "POST":
        values = {gene: request.form.get(gene, "").strip() for gene in FEATURE_COLUMNS}

        if not model_ready:
            error = MODEL_NOT_FOUND_MESSAGE
        else:
            try:
                numeric_values = [float(values[gene]) for gene in FEATURE_COLUMNS]
                input_df = pd.DataFrame([numeric_values], columns=FEATURE_COLUMNS)
                scaled_values = ARTIFACTS["scaler"].transform(input_df)

                encoded_prediction = ARTIFACTS["model"].predict(scaled_values)[0]
                probabilities = ARTIFACTS["model"].predict_proba(scaled_values)[0]

                prediction = ARTIFACTS["label_encoder"].inverse_transform([encoded_prediction])[0]
                probability = round(float(probabilities.max()) * 100, 2)
                result_type = "danger" if prediction.lower() == "abnormal" else "success"
            except ValueError:
                error = "Please enter valid numeric values for all genes."
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        features=FEATURE_COLUMNS,
        values=values,
        prediction=prediction,
        probability=probability,
        result_type=result_type,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
