import constants as const
from tensorflow.keras.utils import plot_model
import json
import matplotlib.pyplot as plt
import numpy as np
# for path, json_file in f.get_json_files_lazily(const.PROCESSED_DATA_DIR):
#
#     x,y = zip(*json_file)5
#     plt.plot(x,y, marker='.')
#     plt.title(path.split('/')[-2] + path.split('/')[-1] )
#     plt.axis('equal')
#     plt.show()

model = const.QUICKDRAW_OCR_MODEL
plot_model(model, to_file='ocr_model.png', show_shapes=True, show_layer_names=True, dpi=96)

with open(const.QUICKDRAW_HISTORY) as f:
    history = json.load(f)

metrics = [k for k in history.keys()]

for metric in metrics:
    plt.plot(history[metric], label=metric)

plt.xticks(np.arange(0, 21, step=1))
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training Metrics')
plt.legend()
plt.grid(True)
plt.show()

def individual_plot(history_file, metric_arg):
    epochs = len(history[metric_arg])
    plt.plot(history_file[metric_arg], label=metric_arg)
    plt.xticks(np.arange(0, epochs, step=1))
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title(f'{metric_arg}')
    plt.grid(True)
    plt.show()

for metric in metrics:
    individual_plot(history, metric)