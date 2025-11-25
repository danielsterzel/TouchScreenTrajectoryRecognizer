import os
from tensorflow.keras.models import load_model
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(BACKEND_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")
PROCESSED_DATA_DIR = os.path.join(BACKEND_DIR, "processed_data")
OCR_MODEL_PATH = os.path.join(MODELS_DIR, 'ocr_model.keras')
OCR_MODEL = load_model(OCR_MODEL_PATH)

ALLOWED_IP_ADDRESSES = ["192.168.40.23", "127.0.0.1", "192.168.40.46", "192.168.40.41"]
SIMPLE_MLP_MODEL = "simple_MLP.keras"