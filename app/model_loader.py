import joblib
import os
import logging
from pathlib import Path

# Cấu hình logger để log lỗi nếu có
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CWD_PROJECT_ROOT = Path(os.getcwd())
RUN_ID_FILE_PATH = CWD_PROJECT_ROOT / 'last_best_run_id.txt'
MODEL_BASE_DIR = CWD_PROJECT_ROOT / 'model'


def get_run_id():
    if not RUN_ID_FILE_PATH.is_file():
        print(f"Debug: Attempting to read run_id from: {RUN_ID_FILE_PATH}")
        raise FileNotFoundError(f"File not found: {RUN_ID_FILE_PATH}")
    try:
        with open(RUN_ID_FILE_PATH, 'r') as f:
            model_id = f.readlines()[-1].strip().split(' - ')[-1]
    except Exception as e:
        print(f"Error reading or parsing run_id from {RUN_ID_FILE_PATH}: {e}")
        raise
    return model_id


def get_model_path():
    """
    Trả về đường dẫn tuyệt đối đến thư mục chứa model đã log từ MLflow.
    Giả định bạn đã biết sẵn run_id của model tốt nhất.
    """
    # TODO: sửa lại run_id này sau khi chọn model

    run_id = get_run_id()

    path1 = f"LGBM/{run_id}/model.pkl"
    path2 = f"XGB/{run_id}/model.pkl"
    path1 = MODEL_BASE_DIR / path1
    path2 = MODEL_BASE_DIR / path2
    abs_path = path1 if os.path.exists(path1) else path2

    return abs_path


def load_model():
    """
    Load model từ thư mục local chứa model MLflow đã log.
    Trả về object model để sử dụng cho dự đoán.
    """
    model_path = get_model_path()
    model = joblib.load(model_path)
    logger.info(f"Model đã được load từ {model_path}")
    return model


def predict(model, features: list):
    """
    Gọi dự đoán từ model với 1 sample feature (dạng list).
    """
    import pandas as pd

    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return prediction[0]
