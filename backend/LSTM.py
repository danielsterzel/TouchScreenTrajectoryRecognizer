from backend import model_related_functions as mrf, constants as const
from ClassifierWrapper import ClassifierWrapper

data, labels = mrf.load_preprocessed_data(const.PROCESSED_DATA_DIR)

file = mrf.return_model_path_if_exists("lstm_model")

if file:
    lstm = ClassifierWrapper()
    lstm.load_model(file, file + "_metadata.json")
    lstm.prepare_data_for_model(data, 0.2)
else:
    lstm = ClassifierWrapper(
        model_type="lstm",
        labels=labels,
        input_shape=(100,2)
    )
    lstm.build_model()
    lstm.prepare_data_for_model(data, test_size=0.2)
    lstm.fit_compile_model(epochs=60, batch_size=16, save_history=True)
    lstm.save_model(filename="lstm_model")
lstm.predict(summary=True)










