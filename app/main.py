from fastapi import FastAPI
from pydantic import BaseModel
from .model_loader import load_model, predict

app = FastAPI()
model = load_model()


class Features(BaseModel):
    data: dict  # Hoặc dùng cụ thể hơn: data: Dict[str, float]


@app.post("/predict")
def predict_endpoint(features: Features):
    input_list = list(features.data.values())  # Hoặc theo đúng thứ tự cột
    result = predict(model, input_list)
    return {"prediction": result}


@app.get("/")
def read_root():
    return {"message": "API is working"}
