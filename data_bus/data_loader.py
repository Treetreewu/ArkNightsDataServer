import json
import os
import re
import data_bus.settings

DATA_PATH = f"../ArknightsGameData/{data_bus.settings.DATA_LANGUAGE}/gamedata/"
DATA_TO_LOAD = {
    "character": "excel/character_table.json",
    "team": "excel/handbook_team_table.json",
}


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
    for c in loaded_data.character.values():
        if c["profession"] != "TOKEN" and c["profession"] != "TRAP":
            print(c["name"])

