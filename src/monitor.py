import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from src.preprocess import clean_text
from src.train import train_model
from src.predict import predict_ufo


def run_drift_report(train_path="data/ufo.csv", new_data_path="data/ufo_new.csv"):
    # --- Load reference and new data ---
    train_df = pd.read_csv(train_path, usecols=["comments"])
    if not os.path.exists(new_data_path):
        # Simulate "new" dataset by sampling from train
        new_df = train_df.sample(frac=0.3, random_state=42)
        new_df.to_csv(new_data_path, index=False)
        print(f"⚠️ No {new_data_path} found, created a simulated new dataset.")
    new_df = pd.read_csv(new_data_path, usecols=["comments"])

    # --- Clean text ---
    train_df["comments"] = train_df["comments"].astype(str).apply(clean_text)
    new_df["comments"] = new_df["comments"].astype(str).apply(clean_text)

    # --- Train/load model ---
    model, vectorizer = train_model(train_path)

    # --- Generate predictions for drift check ---
    train_df["prediction"] = train_df["comments"].apply(lambda x: predict_ufo(x, model, vectorizer))
    new_df["prediction"] = new_df["comments"].apply(lambda x: predict_ufo(x, model, vectorizer))

    # --- Build Evidently report ---
    report = Report(metrics=[
        DataDriftPreset(),   # input data drift
        TargetDriftPreset()  # prediction drift
    ])
    report.run(reference_data=train_df, current_data=new_df)

    with open("reports/drift_report.json", "w", encoding="utf-8") as f:
        f.write(report.as_json())

    # --- Save HTML ---
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/drift_report.html")
    print("Drift + prediction drift report saved to reports/drift_report.html")


if __name__ == "__main__":
    run_drift_report()
