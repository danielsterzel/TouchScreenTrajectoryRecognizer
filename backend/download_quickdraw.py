import requests
import os
import constants as const

CATEGORIES = ["circle", "square", "triangle", "star", "diamond",
 "heart", "lightning", "line", "zigzag", "spiral",
 "arrow", "boomerang", "axe", "bowtie", "cactus",
 "candle", "cloud", "clover", "cup", "door",
 "envelope", "fish", "flower", "house", "ice_cream",
 "key", "ladder", "leaf", "lightbulb", "moon",
 "mountain", "mug", "paper_clip", "snorkel", "snowflake",
 "sun", "sword", "tornado", "tree", "umbrella",
 "vase", "wristwatch", "zebra", "airplane", "bicycle",
 "bus", "car", "cat", "dog", "bird"]


SAVE_DIR = os.path.join(const.DATA_DIR, "quickdraw_npy")
os.makedirs(SAVE_DIR, exist_ok=True)

URL_TEMPLATE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

for category in CATEGORIES:
    url = URL_TEMPLATE.format(category.replace(" ", "_"))
    out_path = os.path.join(SAVE_DIR, f"{category}.npy")

    print("downloading files . . .")

    r = requests.get(url)
    if r.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"Saved: {out_path}")
    else:
        print(f"FAILED: {category} (HTTP {r.status_code})")


