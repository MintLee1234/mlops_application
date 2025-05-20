import mlflow.pyfunc
import os
import logging

# Cấu hình logger để log lỗi nếu có
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_run_id():
    with open('/home/minhle/mlops/last_best_run_id.txt') as f:
        model_id = f.readlines()[-1].strip().split(' - ')[-1]
    return model_id


def get_model_path():
    """
    Trả về đường dẫn tuyệt đối đến thư mục chứa model đã log từ MLflow.
    Giả định bạn đã biết sẵn run_id của model tốt nhất.
    """
    # TODO: sửa lại run_id này sau khi chọn model


    run_id = get_run_id()

    relative_path1 = f"../../mlops/mlruns/614539332736825052/{run_id}/artifacts/model"
    relative_path2 = f"../../mlops/mlruns/621368519697902338/{run_id}/artifacts/model"
    
    abs_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path1))
    abs_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path2))
    abs_path = abs_path1 if os.path.exists(abs_path1) else abs_path2

    return abs_path


def load_model():
    """
    Load model từ thư mục local chứa model MLflow đã log.
    Trả về object model để sử dụng cho dự đoán.
    """
    model_path = get_model_path()
    model = mlflow.pyfunc.load_model(model_path)
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
