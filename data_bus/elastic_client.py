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


class TextMapping:
    index = "text"
    mapping = {
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "name": {"type": "keyword"},
                "tags": {"type": "text"},
            }
        }
    }


class TagMapping:
    index = "tag"
    mapping = {
        "dynamic": "strict",
        "properties": {
            "name": {"type": "keyword"},
            "description": {"type": "text"},
            "itemDesc": {"type": "text"},
            "itemUsage": {"type": "text"},
        }
    }


es = Elasticsearch(HOST)


class DataMapperBase:
    """Do not use this base class directly."""
    index = None
    mapping = None

    @classmethod
    def update_elastic(cls, data: list, remap=False):
        if remap:
            # delete
            cls.delete_all()
            # mapping
            es.indices.create(cls.index, cls.mapping)
        for _id, d in enumerate(data):
            try:
                es.update(cls.index, _id, {"doc": d})
            except Exception:
                es.create(cls.index, _id, d)

        add_tags(wifi_tags)

    @classmethod
    def delete_all(cls):
        es.indices.delete(cls.index)


class TagsMapper(DataMapperBase):
    @classmethod
    def add_tags(cls, tags: dict):
        for wifi in tags:
            if '位移' in tags[wifi] and '控制' not in tags[wifi]:
                tags[wifi].append('控制')
            try:
                es.update_by_query(
                    cls.index,
                    {
                        "query": {"match_phrase": {"name": wifi}},
                        "script": {
                            "inline": f"ctx._source.tags={tags[wifi]}",
                            "lang": "painless",
                            "max_compilations_rate": "500/5m"
                        },
                        "size": 1
                    },
                    conflicts="proceed")
            except:
                print(wifi)
                traceback.print_exc()


class QueryBase:
    """Do not use this base class directly."""
    index = None
    mapping = None

    def __init__(self, plastic_dsl=None, is_query_bool=False):
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
        :param is_query_bool: Whether first param is original elastic dsl bool.
        """
        self.query_bool = {"match_all": {}}
        if plastic_dsl:
            if not is_query_bool:
                print("plastic_dsl:", plastic_dsl)
                plastic_dsl = self.build_query_bool(plastic_dsl)
            print("elastic_dsl:", plastic_dsl)
            self.query_bool = plastic_dsl

    @classmethod
    def get_query_way(cls, prop):
        if prop == "_id":
            return "term"
        prop = cls.mapping.get("mappings").get("properties").get(prop)
        if prop:
            prop_type = prop.get("type")
            if prop_type == "text":
                return "match_phrase"
            else:
                return "term"
        else:
            return "match_phrase"

    @classmethod
    def build_query_bool(cls, plastic_dsl):
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
                    l.append(cls.build_query_bool(ck))
                else:
                    if type(ck[cxk]) == list:
                        l.append({"range": {cxk: {"gte": min(ck[cxk]), "lte": max(ck[cxk])}}})
                    else:
                        l.append({cls.get_query_way(cxk): {cxk: ck[cxk]}})
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
                result = {cls.get_query_way(k): {k: plastic_dsl[k]}}
        return result

    def query(self, size=10000, **kwargs):
        data = es.search(self.index, {
            "query": self.query_bool,
            "size": size,
            **kwargs
        })

        result = {"total": data["hits"]["total"]["value"],
                  "result": []}
        for d in data["hits"]["hits"]:
            d["_source"].update({"id": d["_id"]})
            result["result"].append(d["_source"])
        return result


class TextQuery(QueryBase, TextMapping):
    pass


class TagQuery(QueryBase, TagMapping):
    pass


if __name__ == '__main__':
    # update_elastic(get_update())
    # add_tags(wifi_tags)
    # query(["buff"])
    # print(get_all())
    print(TagQuery().index)
