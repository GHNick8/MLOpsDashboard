import os
os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "false"
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from src.preprocess import load_and_preprocess
import json
import os


# Map numeric labels to human-friendly ones
LABEL_MAP = {
    0: "Explained UFO (likely mundane: plane, star, etc.)",
    1: "Unexplained UFO (strange behavior, unusual)"
}


def train_model(data_path="data/ufo.csv"):
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess(data_path)

    with mlflow.start_run():
        # Train base model
        base_model = LogisticRegression(max_iter=200)
        # Calibrate probabilities for more realistic confidence scores
        model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")

        # Log metrics + params
        mlflow.log_param("model_type", "LogisticRegression+Calibrated")
        mlflow.log_param("max_iter", 200)
        mlflow.log_metric("accuracy", acc)

        # Log classification report as artifact
        clf_report = classification_report(y_test, preds, output_dict=True)
        os.makedirs("reports", exist_ok=True)
        with open("reports/classification_report.json", "w") as f:
            json.dump(clf_report, f, indent=2)
        mlflow.log_artifact("reports/classification_report.json")

        # Example input for model signature
        input_example = vectorizer.transform(["I saw strange lights in the sky"])

        # Log model to MLflow registry
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            registered_model_name="ufo_sightings_classifier"
        )

    return model, vectorizer, LABEL_MAP


if __name__ == "__main__":
    train_model()
