import functools
from collections import defaultdict
import json
import os
import re
from data_bus import settings
from data_bus.elastic_client import es
from elasticsearch import helpers

DATA_PATH = f"../ArknightsGameData/{settings.DATA_LANGUAGE}/gamedata/"

CONTENT_PATTERNS = ['name="(.*?)"', "(PopupDialog).*?", "(Tutorial).*?"]


def remove_blanks(func):
    """Func raw output like:
        [('name1', '', '', 'content1'), ('', '', 'tutorial2', 'content2')]
    return: Like [['name1', 'content1'], ['tutorial2', 'content2']]
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raw = func(*args, **kwargs)
        result = []
        for r in raw:
            name = None
            for i, _ in enumerate(CONTENT_PATTERNS):
                name = name or r[i]
            result.append([name, *r[len(CONTENT_PATTERNS):]])
        return result
    return wrapper


RE_PATTERNS = {
    # remove tags and back slash split line.
    "format": lambda d: re.compile(r"<.*?>|\\\n\s*").sub("", d),
    "content": remove_blanks(re.compile(
        rf'\[(?:{"|".join(CONTENT_PATTERNS)})]\s*(.*)').findall),
    "header": lambda d: re.compile(r"\[HEADER.*?]\s*(.*)").search(d).group(1),
}


def merge_suffix(suffixes, strip="_"):
    """Load data with suffix as the format of:
    {
        suffix1.strip(strip): data1,
        suffix2.strip(strip): data2
    }
    If not any(suffix match), return raw function call.
    """
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(self: BaseLoader, attr_name: str, d, *args, **kwargs):
            for s in suffixes:
                if attr_name.endswith(s):
                    real_name = attr_name[:-len(s)]
                    real_data = self.get(real_name, SimpleObj())
                    if strip:
                        s = s.strip(strip)
                    real_data[s] = func(self, attr_name, d, *args, **kwargs)
                    self[real_name] = real_data
                    break
            else:
                return func(self, attr_name, d, *args, **kwargs)
        return wrapped
    return wrapper


class SimpleObj(defaultdict):
    """An object-like defaultdict(None), can be called either obj.key or
        obj['key']."""
    def __init__(self, d=None):
        super().__init__(None)
        if d:
            self.update(d)

    def __getattr__(self, item):
        return self[item]


class BaseLoader(SimpleObj):
    # TODO: Implement lazy loader.
    def __init__(self, file_list: list):
        super().__init__()
        for file in file_list:
            file_path = os.path.join(DATA_PATH, file)
            if os.path.isfile(file_path):
                self.load_file(file_path)
            elif os.path.isdir(file_path):
                for root, _, files in os.walk(file_path):
                    for f in files:
                        self.load_file(os.path.join(root, f))

    def load_file(self, file_path):
        file_name = os.path.split(file_path)[-1]
        with open(file_path, encoding="utf-8") as file:
            attr_name = self.get_attribute_by_file_name(file_name)
            result = self.load_data(attr_name, file.read())
            if result:
                self[attr_name] = result

    @staticmethod
    def get_attribute_by_file_name(file_name: str):
        # remove extension
        return file_name.split(".")[0]

    def load_data(self, attr_name, d: str):
        """
        return: None or data.
        If return None, Base class believe that load_data of Child class has
        already set data to correct self[attr_name], and will not set again.
        If return data, Base class will set self[attr_name] = data.
        """
        return RE_PATTERNS["format"](d.strip("\n"))


class JsonLoader(BaseLoader):
    """For gamedata/*.json"""
    @staticmethod
    def get_attribute_by_file_name(file_name: str):
        file_name = super(JsonLoader, JsonLoader).get_attribute_by_file_name(file_name)
        if file_name.endswith("_table"):
            file_name = file_name[:-len("_table")]
        return file_name

    def load_data(self, attr_name, d):
        return json.loads(
            super(JsonLoader, self).load_data(attr_name, d),
            object_hook=SimpleObj
        )


class TextLoader(BaseLoader):
    """Plain text loader, for gamedata/story/info/*.txt"""
    def __init__(self, file_list: list):
        super().__init__(file_list)

    @merge_suffix(["_end", "_beg"])
    def load_data(self, attr_name, d: str):
        return super(TextLoader, self).load_data(attr_name, d)


class ConversationLoader(BaseLoader):
    """For gamedata/story/activities/*.txt"""
    @merge_suffix(["_end", "_beg"])
    def load_data(self, attr_name, d: str):
        d = super(ConversationLoader, self).load_data(attr_name, d)
        data = SimpleObj()
        data["header"] = RE_PATTERNS["header"](d)
        data["content"] = RE_PATTERNS["content"](d)
        return data


def _filter_keys(d, reverse=False, key=None, *layered_keys):
    """layered_keys is keys to operate each level."""
    result = SimpleObj()
    for k, v in d.values():
        if (k in layered_keys[0]) ^ reverse:
            if key:
                k = v.get(key, k)
            if isinstance(v, dict) and len(layered_keys) > 1:
                v = _filter_keys(v, reverse, *layered_keys[1:])
            result[k] = v
    return result


def omit(d, key=None, *layered_keys):
    return _filter_keys(d, True, *layered_keys)


def reserve(d, key=None, *layered_keys):
    return _filter_keys(d, False, None, *layered_keys)


def parse_story(story_list):
    return SimpleObj({
        s.storyTitle: sum(s.stories) for s in story_list
    })


def parse_list(d_list, key, *value_keys):
    if len(value_keys) == 0:
        raise ValueError
    result = SimpleObj()
    for d in d_list:
        if len(value_keys) == 1:
            value = d[value_keys[0]]
        else:
            value = SimpleObj({
                k: d[k] for k in value_keys
            })
        result[d[key]] = value
    return result


DATA_TO_LOAD = {
    "json": {
        "loader": JsonLoader,
        "files": [
            "excel/activity_table.json",
            "excel/building_data.json",
            "excel/character_table.json",
            "excel/handbook_team_table.json",
            "excel/range_table.json",
            "excel/charword_table.json"
        ],
    },
    "text": {
        "loader": TextLoader,
        "files": [

        ]
    },
    "conversation": {
        "loader": TextLoader,
        "files": [

        ]
    }
}

# Load data.
loaded_data = SimpleObj({
    "json": JsonLoader(["excel/activity_table.json"])
})



# Organize data

organized_data = SimpleObj({

})


if __name__ == '__main__':
    # for c in loaded_data.character.values():
    #     print(c.name, c.description, c.itemDesc, c.itemUsage)
    DATA_TO_ORGANIZE = SimpleObj({
        "activity": SimpleObj({
            "1stact_zone": parse_list(
                loaded_data.json.activity.activity.DEFAULT["1stact"].zoneList,
                "zoneName", "zoneDesc"
            ),
            "act11d7_camp": reserve(
                loaded_data.json.activity.activity.TYPE_ACT3D0.act11d7.campBasicInfo, None,
                ("campName", "campDesc")
            ),
            "act11d7_story": parse_list(
                loaded_data.json.activity.activity.TYPE_ACT4D0.act7d5.storyInfoList,
                "storyName", "storyDesc"
            ),
            "act13d5_news": reserve(
                loaded_data.json.activity.activity.TYPE_ACT9D0.act13d5.newsInfoList, None,
                ("typeName", "newsTitle", "newsText")
            ),
            "act12d6_buff": reserve(
                loaded_data.json.activity.activity.ROGUELIKE.outBuffInfos, None,
                ("buffUnlockInfos",), ("name", "description", "usage")
            )}),
        "building": SimpleObj({
            "room": reserve(
                loaded_data.json.building_data.rooms, None,
                ("name", "description")
            ),
            "buff": reserve(
                loaded_data.json.building_data.buffs, None,
                ("buffName", "roomType", "description")
            ),
            "furniture": reserve(
                loaded_data.json.building_data.customData.furnitures, None,
                ("name", "usage", "description", "obtainApproach")
            ),
            "theme": reserve(
                loaded_data.json.building_data.customData.themes, None,
                ("name", "desc")
            )
        }),
        "character": reserve(
            loaded_data.json.character, None,
            ("name", "description", "itemUsage", "itemDesc")
        ),
        "line": reserve(
            loaded_data.json.charword, None,
            ("charId", "voiceTitle", "voiceText")
        ),
        "enemy": reserve(
            loaded_data.json.enemy_handbook, None,
            ("name", "enemyRace", "description")
        ),
        "hand_book": SimpleObj({
            h.charID: {
                "drawName": h.drawName,
                "infoName": h.infoName,
                "story": parse_story(h.storyTextAudio)
            } for h in loaded_data.json.handbook_info.handbookDict.values()
        }),
        "item": reserve(
            loaded_data.json.item, "name",
            ("description", "usage")
        ),
        "medal": parse_list(
            loaded_data.json.medal_table.medalList,
            "medalId", "medalName", "description", "getMethod"
        )
    })

    helpers.bulk(es, ({
        '_op_type': 'index',
        '_index': 'info',
        'name': k,
        **v
    } for k, v in DATA_TO_ORGANIZE["info"].items()))
    print("done")
