import collections
import copy
import functools
import os
import re
import traceback

import numpy
import cv2
import intervals
import pytesseract
import platform

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

IMG_COPY_FOR_SHOW = {}


class Config:
    DEBUG = True

    HSV_V_THRESHOLD = 150
    LINES_MERGE_BLUR = 10

    COL_EDGE_INTERVAL = intervals.open(-intervals.inf, -20000 // 920) | intervals.open(40000 // 920, intervals.inf)
    LINE_EDGE_INTERVAL = intervals.open(-intervals.inf, -220000 // 2160) | intervals.open(210000 // 2160,
                                                                                          400000 // 2160)
    LINE_EDGE_RANGE = intervals.open(0.1, 0.7)

    CELL_TEXT_REGION = ((None, None), (493 / 547, 537 / 547))
    CELL_LEVEL_REGION = ((10 / 252, 130 / 252), (437 / 547, 493 / 547))  # ((0, 0), (10, 10))
    CELL_PHASE_REGION = ((42 / 252, 91 / 252), (326 / 547, 402 / 547))


def ocr(image, language="chi_sim+eng", options="--psm 7", **kwargs):
    raw_out = pytesseract.image_to_string(image, language, options, **kwargs)
    return re.sub("/s", "", raw_out)


def cv_imread(file_path, flags=None):
    img = cv2.imread(file_path, flags)
    if img is None:
        img = cv2.imread(file_path.encode('gbk').decode(), flags)
    return img


class Box:
    """
        For a group of images (of each user).
    """

    def __init__(self, images_path):
        self.shape = None
        self._grid_width_gap = None
        self._grid_width = None
        self._grid_height = None
        self._grid_y_starts = None

        self.images = []
        for path, _, files in os.walk(images_path):
            for f in files:
                self.images.append(BoxImage(self, os.path.join(path, f)))
        self.characters = set()

    @property
    def grid_y_starts(self):
        if not self._grid_y_starts:
            for i in self.images:
                try:
                    line_edges = i.find_grid_y()
                    self._grid_height = line_edges[1] - line_edges[0]
                    self._grid_y_starts = [line_edges[0], line_edges[2]]
                    break
                except:
                    traceback.print_exc()
                    pass
        return self._grid_y_starts

    @property
    def grid_height(self):
        if not self._grid_height:
            # call grid_y_starts getter.
            self.grid_y_starts
        return self._grid_height

    @property
    def grid_width(self):
        if not self._grid_width:
            col_edges = []
            for i in self.images:
                try:
                    col_edges.append(i.find_grid_x())
                except:
                    traceback.print_exc()
                    pass
            self._grid_width = find_most_confident([e[1] for e in col_edges])[0]
            self._grid_width_gap = find_most_confident([e[0] for e in col_edges])[0]
        return self._grid_width

    @property
    def grid_width_gap(self):
        if not self._grid_width_gap:
            # call grid_y_starts getter.
            self.grid_width
        return self._grid_width_gap


class BoxImage:
    """
    For one image.
    """

    def __init__(self, box, file_path):
        self.box = box
        self.cells = []
        self._grid_x_starts = None

        self.img_raw = cv_imread(file_path, cv2.IMREAD_COLOR)
        if not self.box.shape:
            self.box.shape = self.img_raw.shape[:2]
        # self.img_gray = cv2.cvtColor(self.img_raw, cv2.COLOR_RGB2GRAY)
        # show(self.img_gray)
        self.img_hsv = cv2.cvtColor(self.img_raw, cv2.COLOR_RGB2HSV)
        # show(self.img_hsv[:, :, 0])
        # show(self.img_hsv[:, :, 1])
        _, self.img_hsv_v_binary = cv2.threshold(self.img_hsv[:, :, 2], Config.HSV_V_THRESHOLD, 255, cv2.THRESH_BINARY)

    def find_grid_y(self):
        """
        For images in one box (a group of images shot on the same device), call only once to set
            box._grid_height and box._grid_y_starts.
        :return: line_edges, length is supposed to be 3.
        """
        if not self.box.grid_height or not self.box.grid_y_starts:
            line_edges = [
                e for e in
                find_edges(self.img_hsv_v_binary, Config.LINE_EDGE_INTERVAL, 1, blank_continues_threshold=0.9)
                if e[0] / self.box.shape[0] in Config.LINE_EDGE_RANGE
            ]
            if len(line_edges) != 3:
                raise Exception(f"Error finding line_edges, too many returns: \n{line_edges}")
            for e in line_edges:
                mark_line(self.img_hsv_v_binary, e[0], 1)

            return [e[0] for e in line_edges]

    def find_grid_x(self):
        """
        :return: most possible [grid_x_gap, grid_x_width] in this image.
        """
        img_hsv_v_binary_core = self.img_hsv_v_binary[
                                self.box.grid_y_starts[0]:self.box.grid_y_starts[1] + self.box.grid_height, :]
        col_edges = find_edges(img_hsv_v_binary_core, Config.COL_EDGE_INTERVAL)
        for e in col_edges:
            mark_line(self.img_hsv_v_binary, e[0])
        result = find_most_confident(slope([e[0] for e in col_edges]))[:2]
        result.sort(key=lambda x: x[0])
        return result

    @property
    def grid_x_starts(self):
        if not self._grid_x_starts:
            pass
        return self._grid_x_starts

    def split_cells(self):
        """
        Must not be called before __init__ done.
        :return:
        """

        show(self.img_hsv_v_binary)


class Cell:
    """
    For one character.
    """

    def __init__(self, box_image, img):
        self.box_image = box_image
        self.img = img


def count_continues(line):
    max_ = 1
    i_ = 0
    for i, pixel in enumerate(line):
        if not i:
            continue
        if pixel != line[i_]:
            max_ = max(i - i_, max_)
            i_ = i
        elif i == len(line) - 1:
            max_ = max(i - i_ + 1, max_)
    return max_


def find_edges(img, interval, axis=0, white_blank=True, blank_continues_threshold=0.3):
    """

    :param blank_continues_threshold:
    :param white_blank: blank area is colored white or black.
    :param img: ndarray
    :param axis:
    :param interval: interval.Interval.
    :return: list of edges found.
    """

    axis_slope = slope(img.sum(axis=axis))
    # mid_index = row_sum.argmax()
    edges = []
    for index, value in enumerate(axis_slope):
        if value / img.shape[axis] in interval:
            # merge close lines as first one.
            if edges and edges[-1][1] ^ value >= 0 and index - edges[-1][0] <= Config.LINES_MERGE_BLUR:
                continue
            # filter edge by continues blank length
            if blank_continues_threshold:
                if value > 0 ^ white_blank:
                    i = index
                else:
                    i = index - 1
                threshold = blank_continues_threshold * img.shape[axis]
                continues = count_continues(img.take([i], axis=1 - axis).flatten())
                if threshold > continues:
                    continue
            edges.append((index, value))
    return edges


def slope(data_list):
    return (0 if i == 0 else int(data_list[i]) - int(data_list[i - 1]) for i, _ in enumerate(data_list))


def find_most_confident(
        data_list, blur_func=lambda x, y: abs(x - y) <= Config.LINES_MERGE_BLUR,
        choose_func=lambda x: round(numpy.average(x))):
    """
    find most typical values in list
    :param data_list:
    :param blur_func: return True to see similar value as one.
    :param choose_func: define which one to choose in all similar values.
    :return: value, confidence
    """
    values = collections.Counter(data_list)
    # sort
    results = list(values.items())
    results.sort(key=lambda x: x[0])
    merged_groups = []
    for i in range(len(results)):
        if not i or not blur_func(results[i][0], results[i - 1][0]):
            merged_groups.append([])
        merged_groups[-1].append(results[i])
    final_results = []
    for group in merged_groups:
        group_value = []
        for g in group:
            group_value.extend([g[0]] * g[1])
        final_results.append((choose_func(group_value), len(group_value)))
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results


def debug(func):
    """
    Debug decorator.
    If not in debug mode, skip the decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if Config.DEBUG:
            return func(*args, **kwargs)

    return wrapper


@debug
def show(img, marked=True, window_text="test"):
    """
    Show image.
    :param marked: mark do not change image itself. If marked, show marked image instead.
    :param img: ndarray
    :param window_text: window name.
    :return:
    """
    if marked:
        copied = IMG_COPY_FOR_SHOW.get(id(img))
        if copied is not None:
            img = copied
    try:
        cv2.imshow(window_text, img)
        cv2.waitKey(0)
    except:
        show(img.img)
        # cv2.waitKey(0)


@debug
def mark(img, area=None, color=128):
    """
    using given color to mark image.
    :param img:
    :param area: [a:b, c:d] as ((a, b), (c, d)).
        For example [:, 1:2] is ((None, None), (1, 2))
    :param color:
    :return:
    """
    if not isinstance(img, numpy.ndarray):
        mark(img.img, area, color)
    else:
        if not area:
            area = ((None, None), (None, None))
        (a, b), (c, d) = area
        copied = IMG_COPY_FOR_SHOW.get(id(img))
        if copied is None:
            IMG_COPY_FOR_SHOW[id(img)] = copied = img.copy()
        copied[a:b, c:d] = color


@debug
def mark_line(img, line_index, axis=0):
    if axis:
        mark(img, ((line_index, line_index + 1), (None, None)))
    else:
        mark(img, ((None, None), (line_index, line_index + 1)))


if __name__ == '__main__':
    my_box = Box("../box_images/test")
    # cv2.destroyAllWindows()
