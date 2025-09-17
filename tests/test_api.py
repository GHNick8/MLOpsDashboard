import os
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_home():
    response = client.get("/v1/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["mode"] in ["local", "mlflow"]

def test_health():
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert "loaded" in data

def test_predict_valid():
    response = client.post("/v1/predict", json={"report": "I saw bright lights in the sky"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data["data"]

def test_predict_empty():
    response = client.post("/v1/predict", json={"report": ""})
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "message" in data
