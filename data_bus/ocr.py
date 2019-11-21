import base64
import math
import os
import re
import requests
from PIL import Image
from data.tags import wifi_tags


TOKEN_API = "https://aip.baidubce.com/oauth/2.0/token"
API = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
API_KEY = "uQiI49m4ha5KooA2WOrS4ZMB"
SECRET_KEY = "0HDGZPdL71KKPlRG0TbO2uAeNSdVEVFq"
INCLUDE_EMP = {"F": "12F", "Lancet": "Lancet-2", "Castle": "Castle-3"}
EXCLUDE_EMP = ["★", "↓", "↑", "*"]
MAX_WIDTH = MAX_HEIGHT = 4096


page_1 = [{'words': '等级稀有度信赖名称'}, {'words': '三↓'}, {'words': '冲★★'}, {'words': '米'}, {'words': '夜莺75'},
                  {'words': '伊芙利特'}, {'words': '蓝毒68'}, {'words': '星熊古米'}, {'words': '55赫默'}, {'words': '55'},
                  {'words': '流星50'}, {'words': '白金50'}, {'words': '可颂'}, {'words': '★'}, {'words': '44艾雅法拉7幽灵鲨'},
                  {'words': '白面鸮'}, {'words': '6)慕斯'}, {'words': '远山'}, {'words': '55'}, {'words': '狮蝎5150火神'},
                  {'words': '蛇屠箱'}, {'words': '50阿米娅'}]


def cleanse(words_list, exclude):
    """
    cleanse employee name list.
    :param words_list: iterable of names.
    :param exclude: exclude key word list.
    :return:
    """
    result = {"result": [], "uncertain": []}
    for phrase in words_list:
        # strip
        phrase = re.match("^[\d()]*(.*?)[\d()]*$", phrase).group()
        # split
        for word in re.split("[\d()]+", phrase):
            if word:
                if word in wifi_tags.keys():
                    result["result"].append(word)
                else:
                    for e in exclude:
                        if e in word:
                            break
                    else:
                        for i in range(len(word)-1):
                            word1 = word[:i+1]
                            word2 = word[i+1:]
                            if word1 in wifi_tags.keys() and word2 in wifi_tags.keys():
                                result["result"].extend([word1, word2])
                                break
                        else:
                            for key in INCLUDE_EMP.keys():
                                if key in word:
                                    result["result"].append(INCLUDE_EMP[key])
                                    break
                            else:
                                result["uncertain"].append(word)
    return result


def merge_images(directory, crop=None, grid=None, delete_source=True):
    """
    merge multiple image files to limited less images
    :param directory: image files directory
    :param crop: crop image by scale before merging. None == (0, 0, 1, 1)
    :param grid: None-auto, tuple-customize. example:(1, 3)
    :param delete_source: delete source image file.
    :return: list of merged image file path.
    """
    image_files = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            try:
                path = os.path.join(root, file)
                image = Image.open(path)
            except:
                continue
            image_format = image.format
            width, height = image.size
            if not crop:
                image_files.append(image)
            else:
                image_files.append(image.crop((crop[0]*width, crop[1]*height, crop[2]*width, crop[3]*height)))
            image.close()
            if delete_source:
                os.remove(path)
    if not image_files:
        return image_files

    # get resolution
    width, height = image_files[0].size
    mode = image_files[0].mode
    if not grid:
        width_count = int(MAX_WIDTH/width)
        height_count = int(MAX_HEIGHT/height)
    else:
        width_count, height_count = grid
    # merge
    result_list = []
    for i in range(math.ceil(len(image_files)/width_count/height_count)):
        result = Image.new(mode, (width*width_count, height*height_count))
        for j in range(width_count):
            for k in range(height_count):
                try:
                    result.paste(image_files.pop(), box=(j*width, k*height))
                except IndexError:
                    break
        result_path = os.path.join(directory, f"merged_{i}.{image_format}")
        result.save(result_path, optimize=True)
        result_list.append(result_path)
    return result_list


def get_access_token(api_key=API_KEY, secret_key=SECRET_KEY):
    """
    get Baidu OCR access_token
    """
    resp = requests.post(TOKEN_API, params={"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key})
    return resp.json().get("access_token")


def ocr(image_list, access_token=None):
    if not access_token:
        access_token = get_access_token()
    # access_token = "24.100c097256f54257cbe61d25b5fa3ea7.2592000.1572077522.282335-11262189"

    words = []
    for image in image_list:
        with open(image, "rb") as png:
            image_str = base64.encodebytes(png.read()).decode()
            resp = requests.post(API,
                                 params={"access_token": access_token},
                                 headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                 data={"image": image_str, "probability": True}).json()
            if resp.get("words_result"):
                words.extend(resp.get("words_result"))
            elif resp.get("error_code") == 110:
                return ocr(image_list, get_access_token())
            else:
                print(resp)
                raise RuntimeError

    words = [word.get("words") for word in words]
    print(words)
    return cleanse(words, EXCLUDE_EMP)


if __name__ == '__main__':
    # print(merge_images("../box_images", (0, 0.4, 1, 1), (1, 3)))
    print(ocr(['../box_images\\merged_0.PNG', '../box_images\\merged_1.PNG'], "W"))



