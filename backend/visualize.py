import matplotlib.pyplot as plt
import functions as f
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, 'processed_data')

for path, json_file in f.get_json_files_lazily(path):

    x,y = zip(*json_file)
    plt.plot(x,y, marker='.')
    plt.axis('equal')
    plt.show()