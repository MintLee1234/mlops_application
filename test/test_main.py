from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import load_model

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():
    input_data = {
        "data": {
            '0': 0.4999999999999999, '1': 0.9921951219512196,
            '2': 0.1044017314980065, '3': 0.1652599462748108,
            '4': 0.2854767990347101, '5': 0.4306741790331501,
            '6': 0.3333333333333335, '7': 0.3478260869565217,
            '8': 0.9999999999998652, '9': 0.0247704644414166,
            '10': 0.5366585411588224, '11': 0.5360931974204285,
            '12': 0.0, '13': 0.5528804815133276,
            '14': 0.5285192469794886, '15': 0.5303415184871578,
            '16': 0.5241852487135507, '17': 0.5276424130398253,
            '18': 0.5316431924882629, '19': 0.5316431924882629,
            '20': 0.6230853391684902, '21': 0.515970515970516
        }
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={"data": {"wrong_feature": 999}}  # Sai format
    )
    assert response.status_code == 422  # Unprocessable Entity


def test_model_predict_directly():
    model = load_model()
    input_data = [[
        0.4999999999999999, 0.9921951219512196, 0.1044017314980065,
        0.1652599462748108, 0.2854767990347101, 0.4306741790331501,
        0.3333333333333335, 0.3478260869565217, 0.9999999999998652,
        0.0247704644414166, 0.5366585411588224, 0.5360931974204285,
        0.0, 0.5528804815133276, 0.5285192469794886, 0.5303415184871578,
        0.5241852487135507, 0.5276424130398253, 0.5316431924882629,
        0.5316431924882629, 0.6230853391684902, 0.515970515970516
    ]]
    prediction = model.predict(input_data)
    assert len(prediction) == 1
