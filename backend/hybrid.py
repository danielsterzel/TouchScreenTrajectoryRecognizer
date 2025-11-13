import constants as const
import model_related_functions as mrf
from ClassifierWrapper import ClassifierWrapper

data, labels = mrf.load_preprocessed_data(const.PROCESSED_DATA_DIR)

hybrid_model = ClassifierWrapper(
    model_type="hybrid_cnn_lstm",
    labels=labels,
)
hybrid_model.build_model()
hybrid_model.prepare_data_for_model(data, 0.2)
hybrid_model.fit_compile_model(epochs=60, batch_size=16, save_history=True)
hybrid_model.save_model(filename="Hybrid_CNN_LSTM")
hybrid_model.predict(summary=True)
