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

CHARACTER_MAPPING = {
    "mappings": {
        "dynamic": "strict",  # 如果遇到新字段抛出异常
        "properties": {
            "char": "char_285_medic2",
            "name": {"type": "keyword"},
            "description": "恢复友方单位生命，且不受<@ba.kw>部署数量</>限制，但再部署时间极长",
            "team": {"type": "byte"},
            "displayNumber": {"type": "keyword"},
            "appellation": {"type": "keyword"},
            "position": {"type": "keyword"},
            "tagList": {"type": "keyword"},
            "itemUsage": "罗德岛医疗机器人Lancet-2，被工程师可露希尔派遣来执行战地医疗任务。",
            "itemDesc": "她知道自己是一台机器人。",
            "itemObtainApproach": "招募寻访",
            "star": 1,
            "profession": "MEDIC",
            "attributesLv1": {
                "type": "object",
                "properties": {
                    "maxHp": {"type": "integer"},
                    "atk": {"type": "integer"},
                    "def": {"type": "integer"},
                    "magicResistance": 0.0,
                    "cost": {"type": "byte"},
                    "blockCnt": {"type": "byte"},
                    "moveSpeed": 1.0,
                    "attackSpeed": 100.0,
                    "baseAttackTime": 2.85,
                    "respawnTime": 200,
                    "hpRecoveryPerSec": 0.0,
                    "spRecoveryPerSec": 1.0,
                    "maxDeployCount": 1,
                    "maxDeckStackCnt": 0,
                    "tauntLevel": 0,
                    "massLevel": 0,
                    "baseForceLevel": 0,
                }
            },
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
        if '位移' in tags[wifi] and '控制' not in tags[wifi]:
            tags[wifi].append('控制')
        try:
            es.update_by_query(INDEX, {"query": {"match_phrase": {"name": wifi}},
                                       "script": {
                                           "inline": f"ctx._source.tags={tags[wifi]}",
                                           "lang": "painless",
                                           "max_compilations_rate": "500/5m"
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
    add_tags(wifi_tags)
    # query(["buff"])
    # print(get_all())
    # print(Query(example).query())
