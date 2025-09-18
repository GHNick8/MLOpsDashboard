🛸 UFO Sightings Classifier (MLOps Project)

An end-to-end MLOps project that classifies UFO sighting reports as Explained (planes, stars, satellites…) or Unexplained UFOs 👽, with model monitoring and a Streamlit dashboard.

This project shows how to take a model from raw data to deployment and monitoring, making it portfolio-ready for both recruiters and ML practitioners.

🚀 Features

Data ingestion & preprocessing

UFO dataset from NUFORC on Kaggle

Text cleaning and vectorization (TF-IDF)

Model training

Logistic Regression with probability calibration

MLflow integration for experiment tracking

Logged accuracy & classification reports

Serving

FastAPI app for predictions

Streamlit dashboard for interactive exploration

Monitoring

Data & prediction drift detection with Evidently

Drift reports saved in JSON + HTML

Monitoring tab in dashboard

Dashboard (Streamlit)

🔍 Single prediction with friendly labels + confidence %

📂 Batch classification with CSV upload + visualization

📊 Monitoring tab with drift summary & full report

⚙️ Tech Stack

Python

Scikit-learn – Model training

MLflow – Experiment tracking & model registry

Evidently – Drift monitoring

FastAPI – Model serving

Streamlit – Interactive dashboard

Pandas / NumPy – Data processing

📂 Project Structure
ufo-mlops/
│── data/                # Raw & processed datasets
│── reports/             # Drift reports (JSON + HTML)
│── src/
│   ├── preprocess.py    # Cleaning & preprocessing
│   ├── train.py         # Model training + MLflow logging
│   ├── predict.py       # Prediction logic
│   ├── monitor.py       # Drift detection with Evidently
│── app.py               # Streamlit dashboard
│── requirements.txt     # Dependencies

▶️ Quickstart

Clone repo & install dependencies:

git clone https://github.com/yourusername/ufo-mlops.git
cd ufo-mlops
pip install -r requirements.txt


Train model:

python -m src.train


Run drift monitoring:

python -m src.monitor


Launch dashboard:

streamlit run app.py

🎯 Future Improvements

Add Not UFO class (to avoid silly predictions on irrelevant text).

Deploy with Docker + cloud service (AWS/GCP/Azure/Replit).

Confidence calibration plots for deeper analysis.

🙌 Why This Project?

This project demonstrates the full MLOps lifecycle:

From data → model → deployment → monitoring → dashboard.

It’s designed to be accessible to both:

Recruiters → clear end-to-end pipeline with visuals.

Engineers → practical use of MLflow, FastAPI, and Evidently.

⚡️ “Not everything is black and white — that’s why the dashboard has a grey background. There’s always a shade of grey when it comes to UFOs.”