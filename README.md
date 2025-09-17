🛸 UFO Sightings Classifier – MLOps Project

An end-to-end MLOps pipeline that classifies UFO sighting reports as Explained or Unexplained.
Built with FastAPI, MLflow, Evidently, Prefect, and Streamlit.

🚀 Features

Data ingestion from Kaggle UFO dataset.

Preprocessing pipeline for cleaning text reports.

Model training with scikit-learn + MLflow experiment tracking.

Model registry (versioned models).

Serving via FastAPI REST API.

Monitoring with Evidently (data drift + prediction drift).

Streamlit dashboard for interactive demo.

📂 Project Structure
ufo-mlops/
│── data/                 # UFO dataset (ignored in git)
│── reports/              # Drift reports (ignored in git)
│── src/                  # Source code
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── monitor.py
│── app.py                # Streamlit dashboard
│── api.py                # FastAPI service
│── requirements.txt
│── README.md
│── .gitignore

⚡ Quickstart

Clone repo:

git clone https://github.com/YOUR_USERNAME/ufo-mlops.git
cd ufo-mlops


Create virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


Run FastAPI service:

uvicorn api:app --reload


Run Streamlit dashboard:

streamlit run app.py

📊 Example

Input:

“Bright lights hovered over the city at night.”

Prediction:
🛸 Unexplained UFO

🔮 Next Steps

Automate retraining with Prefect.

CI/CD with GitHub Actions.

Deploy Streamlit app to Streamlit Cloud or Replit.

💡 This project demonstrates real-world MLOps skills: data ingestion, preprocessing, model training, registry, deployment, monitoring, and visualization.