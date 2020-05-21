import collections
import os
from data_bus.data_loader import loaded_data

class Updater:
    def __init__(self):
        pass

    def update_all(self):
        pass

    def update_character_ocr(self):
        from data_bus.ark_cv_v2 import Config
        chars = "".join(set(i for i in "".join(c["name"] for c in loaded_data.character.values() if c["profession"] != "TOKEN" and c["profession"] != "TRAP")))
        print(chars)
        # with open(os.path.join(Config.TESSERACT_PATH, "tessdata/configs/arknights"), "w") as file:
        #     file.write("tessedit_char_whitelist " + chars)
        return chars


if __name__ == '__main__':
    for c in Updater().update_character_ocr():
        print(c)

