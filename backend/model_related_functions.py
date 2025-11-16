import json
import os
import numpy as np
from backend import constants as const
from tensorflow.keras.models import load_model
from ClassifierWrapper import ClassifierWrapper

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

def get_all_models(model_dir_path):

    models = {}

    for filename in os.listdir(model_dir_path):
        if not filename.endswith(".keras"):
            continue
        saved_model_path = os.path.join(model_dir_path, filename)
        model = ClassifierWrapper()
        model.load_model(saved_model_path, saved_model_path + "_metadata.json")

        stripped_filename = filename.removesuffix(".keras")
        models[stripped_filename] = model
        print(f"Saved model filename: {stripped_filename}")

    return models

def build_and_run_all_models():

    data, labels = load_preprocessed_data(const.PROCESSED_DATA_DIR)
    model_dict = {
        'cnn': '1D_CNN',
        'lstm': 'lstm_model',
        'mlp': 'simple_MLP',
        'hybrid_cnn_lstm': 'hybrid_CNN_LSTM',
    }

    for model_type, model_file in model_dict.items():
        if return_model_path_if_exists(model_file):
            continue
        model = ClassifierWrapper(
            model_type=model_type,
            labels=labels,
            input_shape=(100,2)
        )
        model.build_model()
        model.prepare_data_for_model(data, test_size=0.2)
        model.fit_compile_model(epochs=60, batch_size=16, save_history=True)
        model.save_model(filename=model_file)
        model.predict(summary=True)


def run_all_models():
    models_ = get_all_models(const.MODELS_DIR)
    data, labels = load_preprocessed_data(const.PROCESSED_DATA_DIR)
    # for model in models_.values():
    #     model.prepare_data_for_model(data, test_size=0.2)
    #     model.predict(summary=True)
    for model_name, model in models_.items():
        if model_name == "simple_MLP":
            continue
        model.prepare_data_for_model(data, test_size=0.2)
        model.predict(summary=True)