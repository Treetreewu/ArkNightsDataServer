import base64
import re
from urllib.parse import urlencode, quote

import requests
from elasticsearch import Elasticsearch

from data.tags import wifi_tags

es = Elasticsearch("172.17.11.147")


def build_query(tags: [str], must=True, **kwargs):
    if must:
        word = "must"
    else:
        word = "should"
    result = {
        "query": {
            "bool": {
                word: []
            }
        }
    }
    for t in tags:
        result["query"]["bool"][word].append({"match": {"tags": t}})
    print(result)
    return result


def delete_all():
    es.indices.delete("wifi")


def query(tags):
    print(es.search("wifi", build_query(tags)))


if __name__ == '__main__':
    t = [1, 3]
    f = [45]
    for i in f + t:
        print(i)




