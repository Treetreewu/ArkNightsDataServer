import os
import re
from urllib.parse import unquote

import requests


class ValueUtils:
    @staticmethod
    def to_bool(string, default: bool) -> bool:
        """
        Convert string to bool. String can be None.
        """
        if default:
            if str(string).lower() in ("false", "0", "no", "n"):
                result = False
            else:
                result = True
        else:
            if str(string).lower() in ("true", "1", "yes", "y"):
                result = True
            else:
                result = False
        return result

    @staticmethod
    def to_int(string) -> int or ValueError:
        try:
            return int(string)
        except:
            return None


def download_file(url, path="./", rename_to=None, overwrite=True):
    """
    download file to local path.
    :param rename_to:
    :param url: str
    :param path: save to...
    :param overwrite: bool: when file already exist,
        False: overwrite (default)
        True: rename new file to "file_name_1.*", "file_name_2.*" etc.
    :return: file path
    """
    # connect
    response = requests.get(url)
    # get file name
    cd = response.headers.get("Content-Disposition")
    print(cd)
    if cd:
        # file_name = re.findall("file[-_]?name\s*=\s*(.+)\s*(?:;.*)?$", cd)[0]
        fn = re.findall("filename\*\s*=\s*UTF-8''(.*)[,;\s]?.*?$", cd)
        if not fn:
            fn = re.findall("filename\s*=\s*(.*)[,;\s]?.*?$", cd)
        file_name = unquote(fn[0])
    else:
        file_name = unquote(url.split("/")[-1])

    # create path if not exist
    if not path.endswith("/"):
        path += "/"
    if not os.path.exists(path):
        os.makedirs(path)

    # rename_to
    if "." in file_name:
        file_name1 = "." + file_name.split(".")[-1]
        file_name0 = file_name[:-len(file_name1)]
    else:
        file_name0 = file_name
        file_name1 = ""
    if rename_to:
        file_name0 = rename_to

    file_name = file_name0 + file_name1

    # rename if already exist
    if os.path.exists(path + file_name) and not overwrite:
        # file_name == file_name0 + file_name1
        i = 1
        while os.path.exists(path + file_name0 + " " + str(i) + file_name1):
            i += 1
        file_name = file_name0 + "_" + str(i) + file_name1

    # write
    file = path + file_name
    with open(file, mode='wb+') as f:
        print("saving", file)
        f.write(response.content)
    print("正在下载:" + url)
    return file
