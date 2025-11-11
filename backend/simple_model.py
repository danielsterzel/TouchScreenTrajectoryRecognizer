import os
import numpy as np
from backend import model_related_functions as mrf, constants as const
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


print(const.MODELS_DIR)
data, labels = mrf.load_preprocessed_data(const.PROCESSED_DATA_DIR)
#changes the class names: circle, triangle, rectangle into numeric IDs
# fit learns the mapping and transform applies it so fit_transform does both at once
encoder = LabelEncoder()
label_encoded = encoder.fit_transform(labels)

data_flat = data.reshape((data.shape[0], -1))
# stratify means keep the same proportions in train and test data sets
# e.g. if 30% of samples are circles the test will also have 30%
X_train, X_test, y_train, y_test = train_test_split(data_flat, label_encoded, test_size=0.2, random_state=42, stratify=label_encoded)


model = mrf.return_model_if_exists(const.SIMPLE_MLP_MODEL, directory=const.MODELS_DIR)
if not model:
    model = models.Sequential([
        layers.Input(shape = (X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(np.unique(label_encoded)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))
    model.save(os.path.join(const.MODELS_DIR, const.SIMPLE_MLP_MODEL))

model.summary()
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

correct_classifications_count = np.sum(y_pred == y_test)
print(f"Correct classifications: {correct_classifications_count}")
print(f"Correct classifications proportions: {correct_classifications_count / len(y_test)}")

decoded_y_test = encoder.inverse_transform(y_test)
decoded_y_pred = encoder.inverse_transform(y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(label_encoded))

print(f"Confusion Matrix: \n{conf_matrix}")
print(classification_report(decoded_y_test, decoded_y_pred))
