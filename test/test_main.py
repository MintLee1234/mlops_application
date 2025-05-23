from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import load_model
import numpy as np

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_model_predict_directly():
    model = load_model()
    single_sample_features_1d = [i/100 for i in range(1, 23)]
    input_data_2d = np.array([single_sample_features_1d])
    print(
        f"DEBUG: Shape of input_data_2d for model.predict: "
        f"{input_data_2d.shape}"
    )

    try:
        prediction = model.predict(input_data_2d)
        assert prediction is not None 
        print(f"DEBUG: Prediction result: {prediction}")
    except Exception as e:
        print(f"ERROR during model.predict: {e}")
        if hasattr(model, 'n_features_in_'):
            print(
                "DEBUG: Model expected n_features_in_: "
                f"{model.n_features_in_}"
            )
        if (
            hasattr(model, 'feature_names_in_')
            and model.feature_names_in_ is not None
        ):
            print(f"DEBUG: Model feature_names_in_: {model.feature_names_in_}")
        raise
