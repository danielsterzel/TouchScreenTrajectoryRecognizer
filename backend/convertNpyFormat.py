import numpy as np
import os
import cv2
import constants as const

N_IMAGES_PER_CLASS = 8000

NPPY_DIR = os.path.join(const.DATA_DIR, "quickdraw_npy")
OUT_DIR = os.path.join(const.DATA_DIR, "quickdraw_images")
os.makedirs(OUT_DIR, exist_ok=True)

for folder_name in os.listdir(NPPY_DIR):
    if not folder_name.endswith(".npy"):
        continue

    class_name = folder_name.replace(".npy", "")
    print("Processing:", class_name)

    class_out_dir = os.path.join(OUT_DIR, class_name)
    os.makedirs(class_out_dir, exist_ok=True)

    arr = np.load(os.path.join(NPPY_DIR, folder_name))

    arr = arr[:N_IMAGES_PER_CLASS]

    for index, flat in enumerate(arr):
        img = flat.reshape(28, 28).astype(np.uint8)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        img = 255 - img
        cv2.imwrite(os.path.join(class_out_dir, f"{index}.png"), img)

print("DONE converting all .npy to image folders.")
