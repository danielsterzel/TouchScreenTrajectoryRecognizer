import matplotlib.pyplot as plt
import functions as f
import constants as const

for path, json_file in f.get_json_files_lazily(const.PROCESSED_DATA_DIR):

    x,y = zip(*json_file)
    plt.plot(x,y, marker='.')
    plt.title(path.split('/')[-2] + path.split('/')[-1] )
    plt.axis('equal')
    plt.show()