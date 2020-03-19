import json
import os
import re

DATA_PATH = "../ArknightsGameData/zh_CN/gamedata/"
DATA_TO_LOAD = {
    "character": "excel/character_table.json",
    "team": "excel/handbook_team_table.json",
}

DB_KEYS = []


class LoadedData:
    def __init__(self):
        self.character = {}
        self.team = {}

        for k in DATA_TO_LOAD:
            with open(os.path.join(DATA_PATH, DATA_TO_LOAD[k]), encoding="utf-8") as file:
                file_content = file.read()
                self.__dict__[k] = json.loads(re.sub("<.*?>", "", file_content))


loaded_data = LoadedData()

if __name__ == '__main__':
    print(loaded_data.character.keys())

