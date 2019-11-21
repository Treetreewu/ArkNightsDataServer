import json
import collections
import time
import timeit

from data.tags import TAGS
from data_bus.wiki_spider import get_update

COLLECTION_CACHE = "data/data_collection.json"


def update_collection_cache(data):
    list_set_keys = ['obtain', 'raw_tags']
    set_keys = ['name', 'organization', 'profession', 'gender', 'redeployment', 'attack_speed']
    range_keys = ['stars', 'life', 'attack', 'attack_resistance', 'magic_resistance', 'cost', 'perfect_cost', 'block',]

    collection = {}

    for k in set_keys + list_set_keys:
        collection[k] = set()
    for k in range_keys:
        collection[k] = []

    for wifi in data:
        for k in set_keys:
            collection[k].add(getattr(wifi, k))
        for k in range_keys:
            if len(collection[k]) < 2:
                collection[k].append(getattr(wifi, k))
            else:
                value = getattr(wifi, k)
                collection[k] = [min(collection[k][0], value), max(collection[k][1], value)]
        for k in list_set_keys:
            collection[k].update(getattr(wifi, k))

    collection["infected"] = [True, False]
    collection["tags"] = TAGS

    for k in set_keys + list_set_keys:
        collection[k] = list(collection[k])

    with open(COLLECTION_CACHE, "w", encoding='utf-8') as file:
        json.dump(collection, file, ensure_ascii=False)
    return collection


if __name__ == '__main__':
    update_collection_cache(get_update("../test.html"))

