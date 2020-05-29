from collections import Iterable
import json
import os
import re
import data_bus.settings

DATA_PATH = f"../ArknightsGameData/{data_bus.settings.DATA_LANGUAGE}/gamedata/"
DATA_TO_LOAD = {
    "character": "excel/character_table.json",
    "team": "excel/handbook_team_table.json",
    "range": "excel/range_table.json"
}


class Obj:
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, Iterable) and not isinstance(b, dict):
                setattr(self, a, [Obj(b) if isinstance(b, dict) else b for x in b])
            else:
                setattr(self, a, Obj(b) if isinstance(b, dict) else b)

class LoadedData:
    def __init__(self):
        self.character = {}
        self.team = {}

        for k in DATA_TO_LOAD:
            with open(os.path.join(DATA_PATH, DATA_TO_LOAD[k]), encoding="utf-8") as file:
                file_content = file.read()
                self.__dict__[k] = json.loads(re.sub("<.*?>", "", file_content), object_hook=Obj)


loaded_data = LoadedData()

if __name__ == '__main__':
    print(loaded_data.character)
