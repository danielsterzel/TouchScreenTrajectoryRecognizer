import json
import os
import numpy as np
from backend import constants as const
from tensorflow.keras.models import load_model

def load_preprocessed_data(data_dir):
    data = []
    label = []
    for directory in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, directory)
        if not os.path.isdir(dir_path):
            continue

        for file in os.listdir(dir_path):
            if not file.endswith(".json"):
                continue
            path = os.path.join(dir_path, file)

            with open(path, 'r') as f:
                file_data = json.load(f)

            data.append(np.array(file_data))
            label.append(directory[:-1])


    return np.array(data), np.array(label)

def return_model_path_if_exists(filename, directory=const.MODELS_DIR):
    if not filename.endswith(".keras"):
        filename += ".keras"
    path = os.path.join(directory, filename)
    return path if os.path.exists(path) else None
