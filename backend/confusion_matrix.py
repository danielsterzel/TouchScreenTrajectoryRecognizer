import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import model_related_functions as mrf
import constants as const

X, y, label_to_id, id_to_label = mrf.load_quickdraw_dataset()

model = const.QUICKDRAW_OCR_MODEL

y_pred = model.predict(X, batch_size=32)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y, y_pred_classes)

labels = [id_to_label[str(i)] for i in range(len(id_to_label))]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)

plt.title("Confusion matrix – zbiór testowy")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
