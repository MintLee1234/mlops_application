from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import load_model

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_model_predict_directly():
    model = load_model()
    input_data = [i/100 for i in range(1, 23)]
    prediction = model.predict(input_data)
    assert len(prediction) == 1
