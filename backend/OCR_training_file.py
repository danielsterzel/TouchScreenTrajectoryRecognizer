from OCR import OCR
import tensorflow as tf
import model_related_functions as mrf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import constants as const
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


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

X, y, label_to_id, id_to_label = mrf.load_quickdraw_dataset()

num_classes = len(label_to_id)
y_cat = to_categorical(y, num_classes=num_classes)

with open(os.path.join(const.DATA_DIR, "quickdraw_label_map.json"), "w") as f:
    json.dump(id_to_label, f, indent=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.1, random_state=42, shuffle=True, stratify=y
)

ocr = OCR()
model = ocr.build_model(num_classes=num_classes)

model.compile(optimizer='adam',
              loss= tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy',
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.Precision(name='precision')])

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=callbacks
)
with open(os.path.join(const.MODELS_DIR, "quickdraw_ocr_history.json"), "w") as f:
    json.dump(history.history, f, indent=2)

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title("Accuracy")
plt.savefig(os.path.join(const.MODELS_DIR, "quickdraw_accuracy.png"))
plt.close()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss")
plt.savefig(os.path.join(const.MODELS_DIR, "quickdraw_loss.png"))
plt.close()
# maybe add more plots along the way --- move it to visualize.py also
model.save(os.path.join(const.MODELS_DIR, "quickdraw_ocr_model.keras"))
