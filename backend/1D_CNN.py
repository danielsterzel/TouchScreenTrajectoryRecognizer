import model_related_functions as mrf
import constants as const
from ClassifierWrapper import ClassifierWrapper

data, labels = mrf.load_preprocessed_data(const.PROCESSED_DATA_DIR)

file = mrf.return_model_path_if_exists("1D_CNN")
print(file)

if file:
    cnn_1d = ClassifierWrapper()
    cnn_1d.load_model(file, file + "_metadata.json")
    cnn_1d.prepare_data_for_model(data, 0.2)
else:
    cnn_1d = ClassifierWrapper(
        model_type="cnn",
        labels=labels,
        input_shape=(100,2)
    )
    cnn_1d.build_model()
    cnn_1d.prepare_data_for_model(data, 0.2)
    cnn_1d.fit_compile_model(epochs=60, batch_size=16, save_history=True)
    cnn_1d.save_model(filename="1D_CNN")

cnn_1d.predict(summary=True)
