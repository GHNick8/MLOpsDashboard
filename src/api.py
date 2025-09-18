from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import mlflow.sklearn
import os, json
import streamlit as st

st.markdown(
    """
    <style>
    .stButton button {
        background-color: #00CC88;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 8px 16px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #009966;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------
# Logging Setup
# ---------------------------------------
logger = logging.getLogger("ufo_api")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ---------------------------------------
# Config Management
# ---------------------------------------
env = os.getenv("APP_ENV", "dev")  # default to dev
config_file = os.path.join(os.path.dirname(__file__), "..", "configs", f"config.{env}.json")

with open(config_file, "r") as f:
    config = json.load(f)

MODE = config.get("mode", "local")
MLFLOW_MODEL_URI = config.get("mlflow_model_uri")
LOCAL_DATA_PATH = config.get("local_data_path", "data/ufo.csv")

# ---------------------------------------
# FastAPI App Setup
# ---------------------------------------
app = FastAPI(
    title="UFO Sightings Classifier API",
    description="Classify UFO sighting reports as Explainable or Unexplained.",
    version="1.0.0"
)

# Enable CORS (open for demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# Model Loading
# ---------------------------------------
model, vectorizer = None, None

if MODE == "mlflow":
    try:
        model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
        logger.info(f"Loaded model from MLflow registry: {MLFLOW_MODEL_URI}")
    except Exception as e:
        logger.error(f"Could not load model from MLflow: {e}")
elif MODE == "local":
    from src.train import train_model
    model, vectorizer = train_model(LOCAL_DATA_PATH)
    logger.info(f"Trained model locally from {LOCAL_DATA_PATH}")
else:
    logger.error("Invalid mode in config (use 'mlflow' or 'local')")

# ---------------------------------------
# Schemas & Helpers
# ---------------------------------------
class UFOReport(BaseModel):
    report: str

def success_response(data: dict):
    return {"status": "success", "data": data}

def error_response(message: str, code: int = 400):
    return JSONResponse(status_code=code, content={"status": "error", "message": message})

# ---------------------------------------
# Routes
# ---------------------------------------
@app.get("/v1/")
def home():
    return {"message": "UFO Sightings Classifier API is running!", "mode": MODE}

@app.get("/v1/health")
def health():
    return {
        "status": "ok" if model else "error",
        "mode": MODE,
        "model": "ufo_sightings_classifier",
        "loaded": model is not None
    }

@app.post("/v1/predict", summary="Classify a UFO report",
          description="Submit a UFO sighting description and get back a classification: Explainable or Unexplained.")
def classify(report: UFOReport):
    if not report.report.strip():
        return error_response("Empty report text. Please provide a valid UFO report.", 400)

    if model is None:
        return error_response("No model available. Please retrain or fix config.", 500)

    logger.info(f"Received report: {report.report[:50]}...")
    from src.predict import predict_ufo
    prediction = predict_ufo(report.report, model, vectorizer)
    return success_response({"report": report.report, "prediction": prediction})

@app.get("/")
def root_redirect():
    return {
        "message": "UFO Sightings Classifier API is running!",
        "info": "Use versioned endpoints: /v1/health, /v1/predict"
    }
