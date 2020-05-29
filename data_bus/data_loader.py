from collections.abc import Iterable
from collections import defaultdict
import json
import os
import re
from data_bus import settings

DATA_PATH = f"../ArknightsGameData/{settings.DATA_LANGUAGE}/gamedata/"
DATA_TO_LOAD = {
    "character": "excel/character_table.json",
    "team": "excel/handbook_team_table.json",
    "range": "excel/range_table.json",
    "dialogue": "excel/charword_table.json"
}


class SimpleObj(defaultdict):
    def __init__(self, d):
        super().__init__(None)
        self.update(d)

    def __getattr__(self, item):
        return self[item]


class LoadedData:
    def __init__(self):
        self.character = {}
        self.team = {}

        for k in DATA_TO_LOAD:
            with open(
                    os.path.join(DATA_PATH, DATA_TO_LOAD[k]), encoding="utf-8"
            ) as file:
                file_content = file.read()
                self.__dict__[k] = json.loads(
                    re.sub("<.*?>", "", file_content), object_hook=SimpleObj
                )


loaded_data = LoadedData()

if __name__ == '__main__':
    for c in loaded_data.character.values():
        print(c.name, c.description, c.itemDesc, c.itemUsage)
