import os
import json
import shutil
import numpy as np
import constants as const
from scipy.interpolate import interp1d
import cv2


# def get_next_index(root_dir):
#     existing = [int(file.split('_')[1].split('.')[0]) for file in os.listdir(root_dir)
#                 if file.startswith('points_')]
#     return max(existing, default=0)



def get_json_files_lazily(root_dir):
    for dir_path, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name.endswith(".json"):
                path = os.path.join(dir_path, file_name)
                with open(path, 'r') as f:
                    data = json.load(f)
                yield path, data


def get_x_y_coordinates_from_json(json_file_data, filename=None):
    try:
        return [(p['x'], p['y']) for p in json_file_data]
    except TypeError:
        print(f"Invalid json file: {filename}")
        print("Invalid JSON format:", json_file_data)
        raise


def shift_points_to_0_0(json_file_data):
    if not json_file_data:
        return None
    x0, y0 = json_file_data[0]['x'], json_file_data[0]['y']
    for point in json_file_data:
        point['x'] -= x0
        point['y'] -= y0
    return json_file_data

def resample_trajectory(trajectory_data_from_shifted_json_file, num_of_samples=100, filename=None):
    trajectory_points = np.array(get_x_y_coordinates_from_json(trajectory_data_from_shifted_json_file, filename))
    x, y = trajectory_points[:, 0], trajectory_points[:, 1]

    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cumulative_distance = np.insert(np.cumsum(dist), 0, 0)

    norm = cumulative_distance / cumulative_distance[-1] if cumulative_distance[-1] > 0 else cumulative_distance

    equally_spaced_target_distances = np.linspace(0, 1, num_of_samples)

    fx = interp1d(norm, x)
    fy = interp1d(norm, y)

    x_resampled = fx(equally_spaced_target_distances)
    y_resampled = fy(equally_spaced_target_distances)

    return np.column_stack((x_resampled, y_resampled))


def normalize_points(resampled_trajectory_data):
    x, y = resampled_trajectory_data[:, 0], resampled_trajectory_data[:, 1]
    scale = np.sqrt((x.max() - x.min()) ** 2 + (y.max() - y.min()) ** 2)
    normalized_points = resampled_trajectory_data / scale
    return normalized_points


def preprocess_data_for_model(num_of_samples=100):
    """
    Preprocesses data for AI model:
        1. shifts trajectory points to (0,0)
        2. Resamples trajectory points so that each trajectory has the same number of samples
        3. Saves final data in processed_data folder.
    """
    for json_file_path, json_data in get_json_files_lazily(const.DATA_DIR):
        shifted = shift_points_to_0_0(json_data)
        resampled = resample_trajectory(shifted, num_of_samples)
        normalized_points = normalize_points(resampled)
        if len(normalized_points) == 0:
            print(f"Skipping empty trajectory in {json_file_path}")
            continue

        relative_path = os.path.relpath(str(json_file_path), const.DATA_DIR)  # preserve data folder structure
        new_path = os.path.join(const.PROCESSED_DATA_DIR, relative_path)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        with open(new_path, 'w') as f:
            json.dump(normalized_points.tolist(), f, indent=2)  # type: ignore

    print("All trajectories have been preprocessed and saved in: processed_data folder . . .")

def remove_and_reprocess_data(data_root_dir=const.PROCESSED_DATA_DIR):
    if os.path.exists(data_root_dir):
        print("Removing old processed_data folder")
        shutil.rmtree(data_root_dir)
    print("Reprocessing data . . .")
    preprocess_data_for_model()
    print(f"Saved preprocessed data in: {data_root_dir} folder")

# ----------------- images part -----------------

def get_next_filename(img_dir="data/raw_images"):
    os.makedirs(img_dir, exist_ok=True)
    files = [file for file in os.listdir(img_dir) if file.startswith("drawing")]
    index = len(files)
    return os.path.join(img_dir, f"drawing_{index}.png")


def preprocess_image(img, size):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = img.reshape(size[0], size[1], 1)
    return img

def preprocess_all_images(root_dir=const.IMAGES_DIR, size=(64,64)):
    for dir_path, _ , filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".png"):
                path = os.path.join(dir_path, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Could not read image {path}")
                if img.ndim != 2:
                    raise ValueError(f"Image {path} is not grayscale")

                img = preprocess_image(img, size)

                yield path, img

#