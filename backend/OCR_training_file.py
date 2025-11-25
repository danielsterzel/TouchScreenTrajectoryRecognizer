from OCR import OCR
import numpy as np
import tensorflow as tf
import model_related_functions as mrf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import constants as const
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(const.MODELS_DIR, "ocr_best.keras"),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=2,
        factor=0.5
    )
]

datagen = ImageDataGenerator(rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=10,
    fill_mode='nearest'
)

X, y, kanji_to_index, index_to_kanji = mrf.load_kanji_dataset()

unique, counts = np.unique(y, return_counts=True)

valid_classes = unique[counts >= 3]
mask = np.isin(y, valid_classes)

X = X[mask]
y = y[mask]

print("Filtered to", len(valid_classes), "usable classes.")
print("New dataset size:", X.shape[0])

new_unique = np.unique(y)
new_label_to_id = {old_id: new_idx for new_idx, old_id in enumerate(new_unique)}

y = np.array([new_label_to_id[old] for old in y])

num_classes = len(new_unique)
y_cat = to_categorical(y, num_classes=num_classes)

index_to_kanji_filtered = {
    new_id: index_to_kanji[old_id]
    for old_id, new_id in new_label_to_id.items()
}

with open(os.path.join(const.DATA_DIR, "filtered_class_to_kanji.json"), "w", encoding="utf-8") as f:
    json.dump(index_to_kanji_filtered, f, ensure_ascii=False, indent=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.1, random_state=42, shuffle=True, stratify=y
)

ocr = OCR(input_shape=(64,64,1))
model = ocr.build_model(num_classes=num_classes)

model.compile(optimizer='adam',
              loss= tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=callbacks
)

model.save(os.path.join(const.MODELS_DIR, "ocr_model.keras"))

