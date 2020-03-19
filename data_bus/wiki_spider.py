import os

import requests
from bs4 import BeautifulSoup
import re

from utils import download_file, ValueUtils

COLS = ['image', 'name', 'organization', 'profession', 'stars', 'gender', 'infected', 'obtain', 'life', 'attack',
        'attack_resistance', 'magic_resistance', 'redeployment', 'cost', 'perfect_cost', 'block', 'attack_speed',
        'comment', 'raw_tags']

# ------------------------Deprecated--------------------------

class DoMan:
    """
    Build info entity from bs4 Tags.
    """

    def __init__(self, *args, **kwargs):
        self.image = kwargs.get("image")
        self.name = args[1]
        self.organization = args[2]
        self.profession = args[3]
        self.stars = ValueUtils.to_int(args[4])
        self.gender = args[5]
        self.infected = True if args[6] == "是" else False
        self.obtain = args[7].split("、") if args[7] else []
        self.life = ValueUtils.to_int(args[8])
        self.attack = ValueUtils.to_int(args[9])
        self.attack_resistance = ValueUtils.to_int(args[10])
        self.magic_resistance = ValueUtils.to_int(args[11])
        self.redeployment = args[12]
        self.cost = ValueUtils.to_int(args[13])
        self.perfect_cost = ValueUtils.to_int(args[14]) or self.cost
        # 阻挡数2→3视为3。
        self.block = ValueUtils.to_int(args[15][-1]) if args[15] else 0
        self.attack_speed = args[16]
        self.comment = args[17]
        self.raw_tags = args[18].split("、") if args[17] else []


def get_update(html_cache=None):
    if html_cache:
        with open(html_cache) as file:
            data_html = file.read()
    else:
        data_html = requests.get(
            "http://wiki.joyme.com/arknights/%E5%B9%B2%E5%91%98%E6%95%B0%E6%8D%AE%E8%A1%A8").content.decode()
    soup = BeautifulSoup(data_html)
    wifis = soup.find("table", id="CardSelectTr").find_all("tr")

    ret = []
    for i, w in enumerate(wifis):
        if i == 0:
            continue
        info_list = w.find_all("td")
        do_man = DoMan(*[re.sub("\s", "", i.text) for i in info_list], image=info_list[0].div.div.img["src"])
        if not os.path.exists(f"../static/image/do_man/{do_man.name}.png"):
            download_file(do_man.image, "../static/image/do_man/", rename_to=do_man.name)
        ret.append(do_man)

    return ret



if __name__ == '__main__':
    print(get_update())
