import os
import json
DATASET_DIR = "data/kkanji2"

class_names = sorted(os.listdir(DATASET_DIR))

index_to_unicode = {}
index_to_kanji = {}

for idx, folder in enumerate(class_names):
    unicode_hex = folder[2:]  # remove "U+"
    codepoint = int(unicode_hex, 16)
    kanji_char = chr(codepoint)
    print(f"Unicode for kanji {kanji_char} : {codepoint}\n also quick cheek of the hexagonal value:{unicode_hex}")

    index_to_unicode[idx] = codepoint
    index_to_kanji[idx] = kanji_char

with open('data/class_to_kanji.json', 'w', encoding='utf-8') as f:
    json.dump(index_to_kanji, f, ensure_ascii=False, indent=2)

# with open('data/class_to_unicode.json', 'w', encoding='utf-8') as f:
#     json.dump(index_to_unicode, f, ensure_ascii=False, indent=2)