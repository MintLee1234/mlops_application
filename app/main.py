from fastapi import FastAPI
from model_loader import load_model, predict

app = FastAPI()
model = load_model()

@app.post("/predict")
def predict_endpoint(features: list):
    result = predict(model, features)
    return {"prediction": result}