import json
import os
import numpy as np
from backend import constants as const
from ClassifierWrapper import ClassifierWrapper
import functions as func
import cv2
import requests



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


def load_kanji_dataset(size=(64,64)):
    kanji_dataset_path = os.path.join(const.DATA_DIR, 'kkanji2')

    label_map_path = os.path.join(const.DATA_DIR, "class_to_kanji.json")

    with open(label_map_path) as f:
        label_map = json.load(f)

    index_to_kanji = {int(k): v for k, v in label_map.items()}
    kanji_to_index = {v:k for k, v in index_to_kanji.items()}

    X = []
    y = []

    class_names = sorted(os.listdir(kanji_dataset_path)) # because we built the label map that way
    for idx, folder in enumerate(class_names):
        folder_path = os.path.join(kanji_dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if not file.endswith('.png'):
                continue
            kanji_img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
            kanji_img = func.preprocess_image(kanji_img, size=size)
            X.append(kanji_img)
            y.append(idx)
        print(f"Index of the class:{idx}")
    return np.array(X), np.array(y), kanji_to_index, index_to_kanji


def kanji_predict(img_data, id_to_label):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    # img = func.preprocess_image(img, size=(64,64))
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    # img += noise
    # img = np.clip(img, 0, 255) / 255.0
    # img = np.expand_dims(img, axis=0) # because model outputs batch dimension as well

    #
    #
    # testing this:

    img = 255 - img
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    img = img.astype(np.float32) + noise
    img = np.clip(img, 0, 255) / 255.0
    img = img.reshape(1, 64, 64, 1)

    #
    #
    #

    pred = const.OCR_MODEL.predict(img)
    # cls = int(pred.argmax()) # convert np int to python int
    # unicode_code = id_to_label[cls]
    # kanji = chr(unicode_code)
    # print("id_to_label[cls]:", id_to_label[cls], type(id_to_label[cls]))
    print(f"prediction: {pred}")
    prediction = np.argmax(pred)
    kanji = id_to_label[str(prediction)]
    print(f"returning kanji: {kanji}")

    return kanji

def get_kanji_meaning(kanji):
    url = f"https://jisho.org/api/v1/search/words?keyword={kanji}"
    try:
        r = requests.get(url, timeout=3).json()

        return r["data"][0]["senses"][0]["english_definitions"]
    except Exception as e:
        print(f"Lookup failed for {kanji} error: {e}")
        return ["(no meaning found)"]

