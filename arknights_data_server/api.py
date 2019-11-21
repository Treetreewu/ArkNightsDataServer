import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.views.decorators.cache import cache_page
from django.shortcuts import render

from arknights_data_server.models import User
from data_bus.collection_cacher import update_collection_cache
from data_bus.elastic_client import update_elastic, Query
from data_bus.wiki_spider import get_update


def make_api_response(data, status=200):
    """Make API Response"""
    resp = JsonResponse(data=data, status=status)
    resp['Access-Control-Allow-Origin'] = '*'
    resp['Access-Control-Allow-Methods'] = 'POST'
    resp['Access-Control-Allow-Headers'] = 'Authorization'
    resp['Content-Type'] = 'application/json; charset=utf-8'
    return resp

@require_GET
@cache_page(24*60*60)
def index(request):
    data = Query().query()
    print(data["hits"]["hits"])
    return render(request, "index.html", {"total": data["hits"]["total"]["value"], "wifis": data["hits"]["hits"]})


@csrf_exempt
@require_POST
def search(request):
    json_query = json.loads(request.body)
    return make_api_response(Query(json_query).query())


@csrf_exempt
@require_GET
def all_example(request):

    with open("data/data_collection.json") as file:
        return make_api_response(json.load(file))


@require_POST
def update_wifi(request):
    data = get_update()
    update_elastic(data)
    update_collection_cache(data)



@require_GET
def dsl_search(request):
    from data_bus.elastic_client import es
    # es.search(INDEX, request.GET.json)
    print(request.GET)


@require_http_methods(["PUT"])
def register(request):
    pass
    # User().


def parse_args(request, format):
    if request.method in ["GET", "DELETE"]:
        params = request.GET
    if request.method in ["POST", "PUT"]:
        params = request.POST



