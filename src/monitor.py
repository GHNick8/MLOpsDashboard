import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from src.preprocess import clean_text
from src.train import train_model

def run_drift_report(train_path="data/ufo.csv", new_data_path="data/ufo_new.csv"):
    # --- Load reference and new data ---
    train_df = pd.read_csv(train_path, usecols=["comments"]).sample(2000, random_state=42)  # sample for speed
    if not os.path.exists(new_data_path):
        new_df = train_df.sample(frac=0.3, random_state=42)
        os.makedirs(os.path.dirname(new_data_path), exist_ok=True)
        new_df.to_csv(new_data_path, index=False)
        print(f"No {new_data_path} found, created a simulated new dataset.")
    new_df = pd.read_csv(new_data_path, usecols=["comments"]).sample(1000, random_state=42)

    # --- Clean text ---
    train_df["comments"] = train_df["comments"].astype(str).apply(clean_text)
    new_df["comments"] = new_df["comments"].astype(str).apply(clean_text)

    # --- Train/load model ---
    model, vectorizer, _ = train_model(train_path)

    # --- Generate predictions in batch (labels only) ---
    def batch_predict(texts, model, vectorizer):
        X = vectorizer.transform(texts)
        return model.predict(X)

    train_df["prediction"] = batch_predict(train_df["comments"], model, vectorizer)
    new_df["prediction"] = batch_predict(new_df["comments"], model, vectorizer)

    # --- Build Evidently report ---
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    report.run(reference_data=train_df, current_data=new_df)

    # --- Ensure reports folder ---
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(reports_dir, exist_ok=True)

    # --- Save JSON + HTML ---
    try:
        report.save_json(os.path.join(reports_dir, "drift_report.json"))
    except Exception:
        print("⚠️ JSON export not available in this Evidently version")

    try:
        report.save_html(os.path.join(reports_dir, "drift_report.html"))
    except Exception:
        print("⚠️ HTML export not available in this Evidently version")

    print(f"✅ Drift report saved to {reports_dir}")

if __name__ == "__main__":
    run_drift_report()
