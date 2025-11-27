from backend import model_related_functions as mrf, constants as const
from ClassifierWrapper import ClassifierWrapper

print(const.MODELS_DIR)
data, labels = mrf.load_preprocessed_data(const.PROCESSED_DATA_DIR)
data_flat = data.reshape((data.shape[0], -1))

file = mrf.return_model_path_if_exists("simple_MLP")
if file:
    simple_mlp = ClassifierWrapper()
    simple_mlp.load_model(file, file + "_metadata.json")
    simple_mlp.prepare_data_for_model(data_flat, 0.2)
else:
    simple_mlp = ClassifierWrapper(
        model_type="mlp",
        labels=labels,
        input_shape=(data_flat.shape[1],)
    )
    simple_mlp.build_model()
    simple_mlp.prepare_data_for_model(data_flat,test_size=0.2)

    simple_mlp.fit_compile_model(epochs=30, batch_size=16, save_history=True)
    simple_mlp.save_model(filename=const.SIMPLE_MLP_MODEL)

simple_mlp.predict(summary=True)