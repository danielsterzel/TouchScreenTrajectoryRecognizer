import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(BACKEND_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(BACKEND_DIR, "processed_data")

ALLOWED_IP_ADDRESSES = ["192.168.40.23", "127.0.0.1", "192.168.40.46"]
SIMPLE_MLP_MODEL = "simple_MLP.keras"