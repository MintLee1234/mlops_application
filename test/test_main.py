from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import load_model
import numpy as np

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={"data": {"wrong_feature": 999}}  # Sai format
    )
    assert response.status_code == 422  # Unprocessable Entity


def test_model_predict_directly():
    model = load_model()
    input_data = np.array([[0.1, 0.2, ..., 0.22]])
    prediction = model.predict(input_data)
    assert len(prediction) == 1