import os
from tensorflow.keras.models import load_model
import IP_ADDR as IP_ADDR
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(BACKEND_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")
PROCESSED_DATA_DIR = os.path.join(BACKEND_DIR, "processed_data")
QUICKDRAW_OCR_MODEL_PATH = os.path.join(MODELS_DIR, 'quickdraw_ocr_model.keras')
QUICKDRAW_OCR_MODEL = load_model(QUICKDRAW_OCR_MODEL_PATH)
QUICKDRAW_LABEL_MAP = os.path.join(DATA_DIR, 'quickdraw_label_map.json')
ALLOWED_IP_ADDRESSES = IP_ADDR.ALLOWED_ADDR
SIMPLE_MLP_MODEL = "simple_MLP.keras"