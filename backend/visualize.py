import matplotlib.pyplot as plt
import functions as f
import constants as const
from tensorflow.keras.utils import plot_model
#
# for path, json_file in f.get_json_files_lazily(const.PROCESSED_DATA_DIR):
#
#     x,y = zip(*json_file)5
#     plt.plot(x,y, marker='.')
#     plt.title(path.split('/')[-2] + path.split('/')[-1] )
#     plt.axis('equal')
#     plt.show()

model = const.OCR_MODEL
plot_model(model, to_file='ocr_model.png', show_shapes=True, show_layer_names=True)