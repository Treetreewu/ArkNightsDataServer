import re
import time
import traceback
import requests
from elasticsearch import Elasticsearch

from data_bus.wiki_spider import get_update
from utils import ValueUtils
from data.tags import wifi_tags

HOST = "172.17.11.147"
PORT = 9200
INDEX = "wifi"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "name": {"type": "keyword"},
            "organization": {"type": "keyword"},
            "profession": {"type": "keyword"},
            "stars": {"type": "byte"},
            "gender": {"type": "keyword"},
            "infected": {"type": "boolean"},
            # "obtain": {"type": "array"},
            "life": {"type": "integer"},
            "attack": {"type": "integer"},
            "attack_resistance": {"type": "integer"},
            "magic_resistance": {"type": "integer"},
            "redeployment": {"type": "keyword"},
            "cost": {"type": "byte"},
            "perfect_cost": {"type": "byte"},
            "block": {"type": "byte"},
            "attack_speed": {"type": "keyword"},
            "comment": {"type": "text"},
            # "raw_tags": {"type": "array"},
        }
    }
}

es = Elasticsearch(HOST)


def update_elastic(data, remap=False):
    if remap:
        # delete
        delete_all()
        # mapping
        es.indices.create("wifi", INDEX_MAPPING)
    for index, wifi in enumerate(data):
        wifi_dict = wifi.__dict__
        wifi_dict.pop("image")
        try:
            es.update(INDEX, index, body={"doc": wifi_dict})
        except:
            es.create(INDEX, index, wifi_dict)

    add_tags(wifi_tags)


def add_tags(tags: dict):
    for wifi in tags:
        try:
            es.update_by_query(INDEX, {"query": {"match_phrase": {"name": wifi}},
                                       "script": {
                                           "inline": f"ctx._source.tags={tags[wifi]}",
                                           "lang": "painless",
                                           "max_compilations_rate": ["500/5m"]
                                       },
                                       "size": 1}, conflicts="proceed")
        except:
            print(wifi)
            traceback.print_exc()


def delete_all():
    es.indices.delete(INDEX)


class Query:
    def __init__(self, plastic_dsl=None, query_bool=None):
        """

        :param plastic_dsl: plastic dsl, e.g:
            {
                "AND": [
                    {"stars": 5},
                    {"gender": "女"},
                    {"OR": [
                        {"tags": "buff"},
                        {"magic_resistance": [1, 30]}
                    ]}
                ]
            }
        :param query_bool: elastic dsl bool
        """
        print("plastic_dsl:", plastic_dsl)
        self.query_bool = {"match_all": {}}
        if query_bool:
            self.query_bool = query_bool
        if plastic_dsl.keys():
            self.query_bool = self.build_query_bool(plastic_dsl)

    @staticmethod
    def get_query_way(prop):
        if prop == "_id":
            return "term"
        prop = INDEX_MAPPING.get("mappings").get("properties").get(prop)
        if prop:
            prop_type = prop.get("type")
            if prop_type == "text":
                return "match_phrase"
            else:
                return "term"
        else:
            return "match_phrase"

    @staticmethod
    def build_query_bool(plastic_dsl):
        result = {"bool": {}}
        # k -> key, ck -> child_key, cxk -> child_child_key, etc.
        for k in plastic_dsl:
            l = []
            for ck in plastic_dsl[k]:
                try:
                    cxk = list(ck.keys())[0]
                except AttributeError:
                    # single query without bool.
                    break
                if cxk == "AND" or cxk == "OR" or cxk == "NOT":
                    l.append(Query.build_query_bool(ck))
                else:
                    if type(ck[cxk]) == list:
                        l.append({"range": {cxk: {"gte": min(ck[cxk]), "lte": max(ck[cxk])}}})
                    else:
                        l.append({Query.get_query_way(cxk): {cxk: ck[cxk]}})
            if k == "AND":
                result["bool"]["must"] = l
            elif k == "OR":
                result["bool"]["should"] = l
            elif k == "NOT":
                result["bool"]["must_not"] = l
            else:
                # single query without bool.
                if k == "id":
                    k = "_id"
                result = {Query.get_query_way(k): {k: plastic_dsl[k]}}
        return result

    def query(self, size=10000):
        data = es.search(INDEX, {"query": self.query_bool, "size": size})

        result = {"total": data["hits"]["total"]["value"],
                  "result": []}
        for d in data["hits"]["hits"]:
            d["_source"].update({"id": d["_id"]})
            result["result"].append(d["_source"])
        return result


if __name__ == '__main__':
    update_elastic(get_update())
    # query(["buff"])
    # print(get_all())
    example = {
        "AND": [
            {"stars": [5, 6]},
            {"gender": "女"},
            {"AND": [
                {"tags": "buff"},
                {"profession": "辅助"},
            ]}
        ],
    }
    # print(Query(example).query())
