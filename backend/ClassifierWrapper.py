import numpy as np
from tensorflow.keras import models, layers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List
import constants as const
import os
import json


class ClassifierWrapper:
    available_model_types = ["cnn", "lstm", "mlp", "hybrid_cnn_lstm", "2d_cnn", "ResNet18"]
    EARLY_STOPPER_PUBLIC_PARAMS = {
        "monitor",
        "min_delta",
        "patience",
        "verbose",
        "mode",
        "baseline",
        "restore_best_weights",
        "start_from_epoch"
    }

    def __init__(self, labels=None, model_type=None, input_shape=(100, 2)):

        self.input_shape = input_shape
        self.encoder = LabelEncoder()
        self.labels = labels
        self.model_type = model_type
        self.train_set = None
        self.test_set = None
        if labels is not None:
            self.unique_classes_count = len(np.unique(labels))
            self.output_activation_function = 'sigmoid' if self.unique_classes_count == 2 else 'softmax'
            self.encoded_labels = self.encoder.fit_transform(labels)
            self.loss = 'binary_crossentropy' if self.output_activation_function == 'sigmoid' else 'sparse_categorical_crossentropy'
        else:
            self.unique_classes_count = None
            self.output_activation_function = None
            self.encoded_labels = None
            self.loss = None

        self.metrics = ['accuracy']
        self.early_stoppers = []
        self.model = None
        self.history = None

    def load_model(self, filepath, json_config_path):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")

        if not os.path.isfile(json_config_path):
            raise FileNotFoundError(f"Config file not found: {json_config_path}.")

        self.model = models.load_model(filepath)

        with open(json_config_path, "r") as f:
            json_data = json.load(f)
        for param, value in json_data.items():
            setattr(self, param, value)

        if "class_names" in json_data:
            self.encoder.classes_ = np.array(json_data["class_names"])

        if "labels" in json_data:
            self.labels = np.array(json_data["labels"])
            self.encoded_labels = self.encoder.fit_transform(self.labels)

        history_file_path = filepath +  "history.json"
        if os.path.isfile(history_file_path):
            with open(history_file_path, 'r') as f:
                self.history = json.load(f)

        print(f"Successfully loaded model from {filepath} and it's params from {json_config_path} config file.")

    def _create_mlp(self):
        mlp_model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.unique_classes_count, activation=self.output_activation_function),
        ])
        return mlp_model

    @staticmethod
    def _get_default_callbacks():
        early_stopper = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )

        return [early_stopper, reduce_lr]

    def populate_early_stoppers_with_default(self):
        self.early_stoppers = self._get_default_callbacks()

    def show_early_stoppers(self):
        for early_stopper in self.early_stoppers:
            config = early_stopper.get_config()
            print(config)

    def change_early_stopper_parameter(self, early_stopper, param, value):
        if not hasattr(early_stopper, param):
            raise AttributeError(f"Specified early stopper:{early_stopper} has no attribute:{param}")
        if param not in self.EARLY_STOPPER_PUBLIC_PARAMS:
            raise AttributeError(f"{param} is not a valid parameter")
        setattr(early_stopper, param, value)
        print(f"Updated early stopper:{param} -> {value}")

    def change_all_early_stoppers(self, param, value):
        for callback in self.early_stoppers:
            self.change_early_stopper_parameter(callback, param, value)

    def _create_lstm(self):

        lstm_model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(self.unique_classes_count, activation=self.output_activation_function)
        ])
        return lstm_model

    def _create_1d_cnn(self):

        cnn_1d = models.Sequential([
            layers.Input(shape = self.input_shape),
            layers.Conv1D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.unique_classes_count, activation=self.output_activation_function),
        ])

        return cnn_1d

    def _create_2d_cnn(self):

        self.input_shape = (64,64, 1)
        cnn_2d = models.Sequential([
        layers.Input(shape=self.input_shape),
            # more filters = width of layers = more features learned = more expressive power
            # block 1
            layers.Conv2D(32, kernel_size=3, activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),

            # block 2
            layers.Conv2D(64, kernel_size=3, activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            # block 3
            layers.Conv2D(128, kernel_size=3, activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            layers.Conv2D(64, (1,1), activation='relu'),
            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.unique_classes_count, activation=self.output_activation_function),
        ])
        return cnn_2d

    def _create_res_net_18(self):
        self.input_shape = (64,64, 1)
        res_net_18 = models.Sequential([])
    def _create_hybrid_cnn_lstm(self):
        hybrid = models.Sequential([
            layers.Input(shape = self.input_shape),
            layers.Conv1D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation="relu"),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.unique_classes_count, activation=self.output_activation_function),
        ])

        return hybrid

    def build_model(self):
        match self.model_type:
            case "cnn":
                self.model = self._create_1d_cnn()
            case "lstm":
                self.model = self._create_lstm()
            case "mlp":
                self.model = self._create_mlp()
            case "hybrid_cnn_lstm":
                self.model = self._create_hybrid_cnn_lstm()
            case _:
                raise ValueError(
                    "Invalid model type supported models types are: 1.cnn\n 2.lstm\n 3.mlp\n 4.hybrid_cnn_lstm")

    # def change_model_layers(self, ...):
    #     pass

    def prepare_data_for_model(self, data, test_size=0.2, custom_stratify=None):

        if custom_stratify is None:
            custom_stratify = self.encoded_labels

        x_train, x_test, y_train, y_test = train_test_split(data, self.encoded_labels, stratify=custom_stratify,
                                                            test_size=test_size)

        self.train_set = (x_train, y_train)
        self.test_set = (x_test, y_test)
    def save_model_history(self, history):
        self.history = history

    def fit_compile_model(self, optimizer='adam', epochs=10, batch_size=10, use_encoded_labels=True, save_history=False, callbacks_extend=None):
        if self.model is None:
            raise ValueError("Model has not been built yet")
        if not self.train_set or not self.test_set:
            raise ValueError("Model has no data!\nPlease use the 'prepare_data_for_model()' method first.'")

        original_loss = self.loss
        if not use_encoded_labels and self.loss == 'sparse_categorical_crossentropy':
            self.loss = 'categorical_crossentropy'

        callbacks_to_use = self._get_default_callbacks()
        if callbacks_extend:
            callbacks_to_use.extend(callbacks_extend)

        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        history = self.model.fit(self.train_set[0], self.train_set[1], epochs=epochs, batch_size=batch_size,
                       validation_data=self.test_set, callbacks=callbacks_to_use, verbose=1)
        if save_history:
            self.save_model_history(history)
        self.loss = original_loss



    def populate_metrics(self, metrics: List[str]):
        for metric in metrics:
            if metric not in self.metrics:
                self.metrics.append(metric)

    def print_model_summary(self):
        self.model.summary()

    def bind_class_name_to_encoding(self):
        pass

    def check_predictions(self, class_encoding):
        pass

    def predict(self, x=None, y_test=None, summary=True):
        if x is None:
            if not self.test_set:
                raise ValueError("Cannot predict without any data")
            x = self.test_set[0]
        if y_test is None:
            if not self.test_set:
                raise ValueError("No test data available.")
            y_test = self.test_set[1]

        y_pred_probs = self.model.predict(x)
        if self.output_activation_function == 'sigmoid':
            y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)
        else:
            y_pred = np.argmax(y_pred_probs, axis=1)
        if not hasattr(self.encoder, 'classes_'):
            print("LabelEncoder not fitted. Returning numeric predictions only.")
            return y_pred
        if summary:
            correct_classifications_count_count = np.sum(y_pred == y_test)
            print(f"Correct classifications: {correct_classifications_count_count}")
            print(f"Correct classifications proportions: {correct_classifications_count_count / len(y_test)}")

            decoded_y_test = self.encoder.inverse_transform(y_test)
            decoded_y_pred = self.encoder.inverse_transform(y_pred)
            conf_matrix = confusion_matrix(decoded_y_test, decoded_y_pred, labels=self.encoder.classes_)
            print(f"Confusion Matrix: \n{conf_matrix}")
            print(classification_report(decoded_y_test, decoded_y_pred, zero_division=0))

        return y_pred
    def save_model(self, filename, save_history=True):
        if self.model is None:
            raise ValueError("No model to save")
        os.makedirs(const.MODELS_DIR, exist_ok=True)
        if not filename.endswith(".keras"):
            filename += ".keras"
        model_path = os.path.join(const.MODELS_DIR, filename)
        metadata_path = os.path.join(const.MODELS_DIR, f"{filename}_metadata.json")
        history_path = os.path.join(const.MODELS_DIR, f"{filename}_history.json")

        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        metadata = {
            "model_type": self.model_type,
            "input_shape": self.input_shape,
            "unique_classes_count": self.unique_classes_count,
            "class_names": self.encoder.classes_.tolist(),
            "metrics": self.metrics,
            "loss": self.loss,
            "labels": self.labels.tolist() if isinstance(self.labels, np.ndarray) else self.labels,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4) # type: ignore
        print(f"Model metadata saved to: {metadata_path}")

        if save_history and self.history is not None:
            with open(history_path, "w") as f:
                json.dump(self.history.history, f, indent=4) # type: ignore
            print(f"Training history saved to: {history_path}")