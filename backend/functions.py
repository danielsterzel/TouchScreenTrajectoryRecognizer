import os
import json
import numpy as np
from scipy.interpolate import interp1d

def get_next_index(root_dir):
    existing = [int(file.split('_')[1].split('.')[0]) for file in os.listdir(root_dir)
                if file.startswith('points_')]
    return max(existing, default=0)

def get_json_files_lazily(root_dir):
    for dir_path, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name.endswith(".json"):
                path = os.path.join(dir_path, file_name)
                with open(path, 'r') as f:
                    data = json.load(f)
                yield path, data

def shift_points_to_0_0(root_dir):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "shifted_data")

    for json_file_path,json_file_data in get_json_files_lazily(root_dir):
        if not json_file_data:
            continue
        x0, y0 = json_file_data[0]['x'], json_file_data[0]['y']
        for point in json_file_data:
            point['x'] -= x0
            point['y'] -= y0

        relative_path = os.path.relpath(str(json_file_path), root_dir) # preserve data folder structure
        new_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        with open(new_path, 'w') as f:
            json.dump(json_file_data, f, indent=2)  # type: ignore

def get_x_y_coordinates_from_json(json_file_data):
    points = [(p['x'], p['y']) for p in json_file_data]
    return points

def resample_trajectory(trajectory_data_from_shifted_json_file, num_of_samples = 100):
    trajectory_points = np.array(get_x_y_coordinates_from_json(trajectory_data_from_shifted_json_file))
    x, y = trajectory_points[:, 0], trajectory_points[:, 1]

    dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
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
    scale = np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2)
    normalized_points = resampled_trajectory_data / scale
    return normalized_points