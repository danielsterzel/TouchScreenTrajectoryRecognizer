import json
import os
import numpy as np
from backend import constants as const
from ClassifierWrapper import ClassifierWrapper
import cv2

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
    for model_name, model in models_.items():
        if model_name == "simple_MLP":
            continue
        model.prepare_data_for_model(data, test_size=0.2)
        model.predict(summary=True)

def load_quickdraw_dataset(size=(64, 64), max_per_class=1000):
    dataset_dir = os.path.join(const.DATA_DIR, "quickdraw_images")

    X = []
    y = []

    class_names = sorted(
        directory for directory in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, directory))
    )

    label_to_id = {class_name: id_ for id_, class_name in enumerate(class_names)}
    id_to_label = {str(id_): class_name for class_name, id_ in label_to_id.items()}

    print(f"Detected {len(class_names)} classes:")
    print(class_names)

    for class_name in class_names:
        class_id = label_to_id[class_name]
        class_dir = os.path.join(dataset_dir, class_name)

        print(f" Loading class: {class_name}")
        count = 0

        for filename in os.listdir(class_dir):
            if not filename.endswith(".png"):
                continue

            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f" Could not read {img_path}.")
                continue

            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            img = img.astype("float32") / 255.0
            img = img.reshape(size[0], size[1], 1)

            X.append(img)
            y.append(class_id)
            count += 1

            if count >= max_per_class:
                break

    X = np.array(X)
    y = np.array(y)
    print("Dataset loaded . . .")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y, label_to_id, id_to_label

def quickdraw_predict_img(img_data):

    img = cv2.imdecode(np.frombuffer(img_data, np.uint8),  cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Couldn't decode the image")
    img = 255 - img

    _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(img)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    side = max(w, h)
    square = 255 * np.ones((side, side), dtype=np.uint8)
    offset_x = (side - w) // 2
    offset_y = (side - h) // 2
    square[offset_y:offset_y + h, offset_x:offset_x + w] = cropped

    img = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(const.PROCESSED_DATA_DIR, 'processed_img.png'), img)
    img = img.astype(np.float32) / 255.0

    img = img.reshape(64, 64, 1)
    img = np.expand_dims(img, axis=0)
    model = const.QUICKDRAW_OCR_MODEL

    class_probabilities = model.predict(img)
    classification = int(np.argmax(class_probabilities))

    with open(const.QUICKDRAW_LABEL_MAP, "r") as f:
        quickdraw_label_map = json.load(f)

    return quickdraw_label_map[str(classification)]