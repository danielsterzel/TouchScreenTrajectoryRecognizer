import os
import json

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
                yield data

def shift_points_to_0_0(root_dir):
    for trajectory in get_json_files_lazily(root_dir):
        print(len(trajectory), "points")