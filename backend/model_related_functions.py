import json
import os
import numpy as np
from backend import constants as const
import tensorflow as tf
from tensorflow.keras.models import load_model
from ClassifierWrapper import ClassifierWrapper
import deeplake
import functions as func
import pickle
import cv2


CACHE_FILE = os.path.join(const.PROCESSED_DATA_DIR,"cached_kanji_dataset.npz") # change to processed data dir
LABEL_MAP_FILE = os.path.join(const.PROCESSED_DATA_DIR,"cached_label_maps.pkl") # change to processed data dir
OCR_MODEL_PATH = os.path.join(const.MODELS_DIR, 'ocr_model.keras')
OCR_MODEL = tf.keras.models.load_model(OCR_MODEL_PATH)

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



def load_kanji_dataset(size=(64,64), use_cache=True):
    if use_cache and os.path.exists(CACHE_FILE) and os.path.exists(LABEL_MAP_FILE):
        print("Loading cached dataset...")
        data = np.load(CACHE_FILE)
        X = data["X"]
        y = data["y"]

        with open(LABEL_MAP_FILE, "rb") as f:
            label_to_id, id_to_label = pickle.load(f)

        return X, y, label_to_id, id_to_label

    print("Cache not found. Loading and preprocessing DeepLake dataset...")
    ds = deeplake.load("hub://activeloop/kuzushiji-kanji")

    print("Collecting labels...")
    all_labels = [sample["labels"].numpy().item() for sample in ds]

    unique_labels = np.unique(all_labels)
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    X = []
    y = []

    print("Processing images...")
    length = len(ds)
    for i, sample in enumerate(ds):
        print(f"Iteration {i}/{length}")
        img = sample["images"].numpy()
        lbl = sample["labels"].numpy().item()

        img = func.preprocess_image(img, size)

        X.append(img)
        y.append(label_to_id[lbl])

    X = np.array(X)
    y = np.array(y)

    print("Saving dataset to cache...")
    np.savez_compressed(CACHE_FILE, X=X, y=y)

    with open(LABEL_MAP_FILE, "wb") as f:
        pickle.dump((label_to_id, id_to_label), f)

    print("Cache saved!")
    return X, y, label_to_id, id_to_label

def kanji_predict(img_data, id_to_label):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = func.preprocess_image(img, size=(64,64))
    img = np.expand_dims(img, axis=0) # because model outputs batch dimension as well

    pred = OCR_MODEL.predict(img)
    cls = pred.argmax()
    return id_to_label[cls]
