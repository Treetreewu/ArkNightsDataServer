import collections
import copy
import functools
import os
import re
import traceback
import numpy
import cv2
import intervals
import locale
import pytesseract
import ctypes
from pyocr.libtesseract import tesseract_raw

IMG_COPY_FOR_SHOW = {}


class Config:
    DEBUG = True
    C_TESS_MODE = True
    TESSERACT_PATH = "C:\\Program Files\\Tesseract-OCR"
    TESSERACT_DATA_PATH = os.path.join(TESSERACT_PATH, "tessdata")
    TESSERACT_LIB = [os.path.join(TESSERACT_PATH, "libtesseract-5.dll"), "libtesseract.so"]
    TEXT_OCR_CONFIGS = ["load_system_dawg 0", "load_freq_dawg 0", "user_words_suffix user-words", "user_patterns_suffix user-patterns"]
    HSV_V_THRESHOLD = 150
    HSV_V_TEXT_THRESHOLD = 150
    GRAY_THRESHOLD = 135
    LINES_MERGE_BLUR = 10
    OCR_RESIZE = 1.2

    COL_EDGE_INTERVAL = intervals.open(-intervals.inf, -20000 // 920) | intervals.open(40000 // 920, intervals.inf)
    LINE_EDGE_INTERVAL = intervals.open(-intervals.inf, -200000 // 2160) | intervals.open(210000 // 2160,
                                                                                          400000 // 2160)
    LINE_EDGE_RANGE = intervals.open(0.1, 0.7)

    CELL_TEXT_REGION = ((396 / 442, 434 / 442), (20 / 203, None))
    CELL_LEVEL_REGION = ((351 / 442, 398 / 442), (17 / 203, 87 / 203))  # ((0, 0), (10, 10))
    CELL_PHASE_REGION = ((264 / 442, 322 / 442), (34 / 203, 72 / 203))


pytesseract.pytesseract.tesseract_cmd = os.path.join(Config.TESSERACT_PATH, "tesseract.exe")


class Tesseract:
    # Singleton. Change of params result to renew object.
    _instance = None
    _init_args = None
    _init_kwargs = None

    def __new__(cls, *args, **kwargs):
        if cls._instance and (cls._init_kwargs is not None and cls._init_kwargs == kwargs)\
                         and (cls._init_args is not None and cls._init_args == args):
            pass
        else:
            cls._init_args = args
            cls._init_kwargs = kwargs
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, language=None, oem=1, configs=None):
        """
        init TessBaseAPI with TessBaseAPIInit1.
        :param language:
        :param oem:
        :param configs:
        """
        if configs is None:
            configs = Config.TEXT_OCR_CONFIGS

        # add libnames
        if not set(Config.TESSERACT_LIB).issubset(tesseract_raw.libnames):
            tesseract_raw.libnames.extend(Config.TESSERACT_LIB)
        print(tesseract_raw.libnames)
        # reload libs
        for libname in tesseract_raw.libnames:  # pragma: no branch
            try:
                tesseract_raw.g_libtesseract = ctypes.cdll.LoadLibrary(libname)
                tesseract_raw.lib_load_errors = []
                break
            except OSError as ex:  # pragma: no cover
                if hasattr(ex, 'message'):
                    # python 2
                    tesseract_raw.lib_load_errors.append((libname, ex.message))
                else:
                    # python 3
                    tesseract_raw.lib_load_errors.append((libname, str(ex)))

        assert tesseract_raw.g_libtesseract

        # Tesseract 4 workaround
        if tesseract_raw.get_version() == "4.0.0":
            locale.setlocale(locale.LC_ALL, "C")

        handle = tesseract_raw.g_libtesseract.TessBaseAPICreate()
        try:
            if language:
                language = language.encode("utf-8")
            prefix = Config.TESSERACT_DATA_PATH.encode("utf-8")
            # if tesseract_raw.TESSDATA_PREFIX:  # pragma: no cover
            #     prefix = tesseract_raw.TESSDATA_PREFIX.encode("utf-8")
            tesseract_raw.g_libtesseract.TessBaseAPIInit1(
                ctypes.c_void_p(handle),
                ctypes.c_char_p(prefix),
                ctypes.c_char_p(language),
                ctypes.c_int(oem),
                (ctypes.POINTER(ctypes.c_char_p) * len(configs))(*[ctypes.c_char_p(c) for c in configs]),
                ctypes.c_int(len(configs))
            )
            tesseract_raw.g_libtesseract.TessBaseAPISetVariable(
                ctypes.c_void_p(handle),
                b"tessedit_zero_rejection",
                b"F"
            )
        except:  # noqa: E722
            tesseract_raw.g_libtesseract.TessBaseAPIDelete(ctypes.c_void_p(handle))
            raise
        self.handle = handle

    def __del__(self):
        try:
            tesseract_raw.cleanup(self.handle)
        except AttributeError:
            pass

    def ocr(self, img, psm=7, digits=False):
        tesseract_raw.set_page_seg_mode(self.handle, psm)
        tesseract_raw.set_is_numeric(self.handle, digits)
        tesseract_raw.set_image(self.handle, img)
        text = tesseract_raw.get_utf8_text(self.handle)
        tesseract_raw.cleanup(self.handle)
        return text




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


def ocr(image, language="chi_sim", oem=1, psm=7, options="", c_tess_mode=Config.C_TESS_MODE, **kwargs):
    if isinstance(image, ImageBase):
        image = image.img
    pretreated = 255 - image
    ImageBase(pretreated).show()

    if c_tess_mode:
        tesseract = Tesseract(language, oem)
        digits = False
        if "digits" in options:
            digits = True
        raw_out = tesseract.ocr(image, psm, digits)
    else:
        raw_out = pytesseract.image_to_string(pretreated, language, f"--oem {oem} --psm {psm} {options}", **kwargs)
    return re.sub("\s", "", raw_out)


def cv_imread(file_path, flags=None):
    img = cv2.imread(file_path, flags)
    if img is None:
        img = cv2.imread(file_path.encode('gbk').decode(), flags)
    return img


class ImageBase:
    def __init__(self, img, *args, **kwargs):
        self.img = img

    def slice(self, area=None, img=None):
        """
        :param img: specify image to use. If None, use self.img
        :param area: percent of area.
        :return:
        """
        if img is None:
            img = self.img
        if not area:
            area = ((None, None), (None, None))
        (a, b), (c, d) = area
        edges = [a, b, c, d]
        multiple = [img.shape[0]] * 2 + [img.shape[1]] * 2
        for index, e in enumerate(edges):
            if e:
                edges[index] = e * multiple[index]
        edges = list(map(lambda x: round(x) if x else None, edges))
        return img[edges[0]:edges[1], edges[2]:edges[3]]

    @debug
    def show(self, marked=True, window_text="test"):
        """
        Show image.
        :param marked: mark do not change image itself. If marked, show marked image instead.
        :param window_text: window name.
        :return:
        """
        img = self.img
        if marked:
            copied = IMG_COPY_FOR_SHOW.get(id(self))
            if copied is not None:
                img = copied
        cv2.imshow(window_text, img)
        cv2.waitKey(0)
        return self

    @debug
    def mark(self, area=None, color=128):
        """
        using given color to mark image.
        :param area: [a:b, c:d] as ((a, b), (c, d)).
            For example [:, 1:2] is ((None, None), (1, 2))
        :param color:
        :return:
        """
        if not area:
            area = ((None, None), (None, None))
        (a, b), (c, d) = area
        copied = IMG_COPY_FOR_SHOW.get(id(self))
        if copied is None:
            IMG_COPY_FOR_SHOW[id(self)] = copied = self.img.copy()
        copied[a:b, c:d] = color
        return self

    @debug
    def mark_line(self, line_index, axis=0):
        if axis:
            return self.mark(((line_index, line_index + 1), (None, None)))
        else:
            return self.mark(((None, None), (line_index, line_index + 1)))

    def save(self, filename="test.tif", path="../box_images/saved"):
        cv2.imencode("." + filename.split(".")[-1], self.img)[1].tofile(os.path.join(path, filename))
        # cv2.imwrite(os.path.join(path, filename), self.img)
        print(f"{os.path.join(path, filename)} saved")
        return self

    def find_edges(self, interval, axis=0, white_blank=True, blank_continues_threshold=0.3):
        """
        :param blank_continues_threshold:
        :param white_blank: blank area is colored white or black.
        :param axis:
        :param interval: interval.Interval.
        :return: list of edges found.
        """
        axis_slope = slope(self.img.sum(axis=axis))
        # mid_index = row_sum.argmax()
        edges = []
        for index, value in enumerate(axis_slope):
            if value / self.img.shape[axis] in interval:
                # merge close lines as first one.
                if edges and edges[-1][1] ^ value >= 0 and index - edges[-1][0] <= Config.LINES_MERGE_BLUR:
                    continue
                # filter edge by continues blank length
                if blank_continues_threshold:
                    if value > 0 ^ white_blank:
                        i = index
                    else:
                        i = index - 1
                    threshold = blank_continues_threshold * self.img.shape[axis]
                    continues = count_continues(self.img.take([i], axis=1 - axis).flatten())
                    if threshold > continues:
                        continue
                edges.append((index, value))
        return edges


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
                img = cv_imread(os.path.join(path, f), cv2.IMREAD_COLOR)
                self.images.append(BoxImage(self, img))
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
                    col_edges.append(i.find_grid_width())
                except:
                    traceback.print_exc()
                    pass
            self._grid_width = int(find_most_confident([e[1] for e in col_edges])[0][0])
            self._grid_width_gap = int(find_most_confident([e[0] for e in col_edges])[0][0])
        return self._grid_width

    @property
    def grid_width_gap(self):
        if not self._grid_width_gap:
            # call grid_y_starts getter.
            self.grid_width
        return self._grid_width_gap


class BoxImage(ImageBase):
    """
    For one image.
    """

    def __init__(self, box, img, *args, **kwargs):
        self.img_raw = img
        self.img_gray = cv2.cvtColor(self.img_raw, cv2.COLOR_RGB2GRAY)
        _, self.img_gray_binary = cv2.threshold(self.img_gray, Config.GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)
        self.img_hsv = cv2.cvtColor(self.img_raw, cv2.COLOR_RGB2HSV)
        _, self.img_hsv_v_binary = cv2.threshold(self.img_hsv[:, :, 2], Config.HSV_V_THRESHOLD, 255, cv2.THRESH_BINARY)
        super().__init__(self.img_hsv_v_binary, *args, **kwargs)

        self.box = box
        self._cells = []
        self._grid_x_starts = None
        self._col_edges = None

        if not self.box.shape:
            self.box.shape = self.img.shape[:2]

    def find_grid_y(self):
        """
        For images in one box (a group of images shot on the same device), call only once to set
            box._grid_height and box._grid_y_starts.
        :return: line_edges, length is supposed to be 3.
        """
        line_edges = [
            e for e in
            self.find_edges(Config.LINE_EDGE_INTERVAL, 1, blank_continues_threshold=0.8)
            if e[0] / self.box.shape[0] in Config.LINE_EDGE_RANGE
        ]
        if len(line_edges) != 3:
            raise Exception(f"Error finding line_edges, too many returns: \n{line_edges}")
        for e in line_edges:
            self.mark_line(e[0], 1)

        return [e[0] for e in line_edges]

    def find_grid_width(self):
        """
        find grid_x_gap, grid_x_width
        :return: most possible [grid_x_gap, grid_x_width] in this image.
        """
        img_hsv_v_binary_core = self.img_hsv_v_binary[
                                self.box.grid_y_starts[0]:self.box.grid_y_starts[1] + self.box.grid_height, :]
        self._col_edges = ImageBase(img_hsv_v_binary_core).find_edges(Config.COL_EDGE_INTERVAL)
        for e in self._col_edges:
            self.mark_line(e[0])
        result = find_most_confident(slope(self._col_edges, lambda x: x[0]))[:2]
        result.sort(key=lambda x: x[0])
        return [r[0] for r in result]

    @property
    def grid_x_starts(self):
        if not self._grid_x_starts:
            for index, e in enumerate(slope(self.col_edges, key=lambda x: x[0])):
                if e == self.box.grid_width_gap:
                    seed = self.col_edges[index][0]
                    break
            else:
                raise Exception("grid_x_starts seed not found.")
            distance = self.box.grid_width + self.box.grid_width_gap
            self._grid_x_starts = [int(seed + x * distance) for x in range(
                -int(seed / distance), int((self.img_hsv_v_binary.shape[1] - seed) / distance))]

        return self._grid_x_starts

    @property
    def col_edges(self):
        if not self._col_edges:
            self.find_grid_width()
        return self._col_edges

    @property
    def cells(self):
        """
        Must not be called before __init__ done.
        :return:
        """
        if not self._cells:
            for x_start in self.grid_x_starts:
                for y_start in self.box.grid_y_starts:
                    self._cells.append(Cell(self, self.img_raw[y_start:y_start + self.box.grid_height,
                                                               x_start:x_start + self.box.grid_width]))
            # show(self.img_hsv_v_binary)
        return self._cells


COUNT = 0


class Cell(ImageBase):
    """
    For one character.
    """

    def __init__(self, box_image, img, *args, **kwargs):
        self.img_raw = img
        self.img_hsv = cv2.cvtColor(self.img_raw, cv2.COLOR_RGB2HSV)
        _, self.img_hsv_v_binary = cv2.threshold(self.img_hsv[:, :, 2], Config.HSV_V_THRESHOLD, 255, cv2.THRESH_BINARY)
        super().__init__(self.img_hsv_v_binary, *args, **kwargs)
        self.box_image = box_image
        self._name = None
        self._phase = None
        self._level = None

    @property
    def name(self):
        if not self._name:
            text_area = self.slice(Config.CELL_TEXT_REGION, self.img_raw)
            text_area = cv2.resize(text_area, (round(60 / text_area.shape[0] * text_area.shape[1]), 60),
                                   interpolation=cv2.INTER_LINEAR)
            text_area_v = cv2.cvtColor(text_area, cv2.COLOR_RGB2HSV)[:, :, 2]
            _, img_for_text_ocr = cv2.threshold(text_area_v, Config.HSV_V_TEXT_THRESHOLD, 255, cv2.THRESH_BINARY)

            self._name = ocr(img_for_text_ocr, options='arknights', nice=1)
        return self._name

    @property
    def level(self):
        if not self._level:
            self._level = ocr(self.slice(Config.CELL_LEVEL_REGION), None, options="digits")
        return self._level

    @property
    def phase(self):
        if not self._phase:
            weight = {
                0: 0,
                1: 0,
                2: 0
            }
            phase_img = self.slice(Config.CELL_PHASE_REGION)
            col_half_index = phase_img.shape[1]//2
            col_2_change = sum(abs(x) for x in slope(phase_img[:, 1]))//255
            col_half_change = sum(abs(x) for x in slope(phase_img[:, col_half_index]))//255
            col_half_continues_rate = count_continues(phase_img[:, col_half_index]) / phase_img.shape[0]

            if col_2_change <= 1:
                weight[0] += 1
            elif col_2_change <= 4:
                weight[1] += 1
            else:
                weight[2] += 1

            if col_half_change == 4:
                weight[2] += 1
            elif col_half_change == 2:
                weight[1] += 1
            else:
                weight[0] += 1

            if col_half_continues_rate > 0.5:
                if phase_img[phase_img.shape[0]*2//5, col_half_index] > 128:
                    weight[0] += 1
                else:
                    weight[1] += 1
            elif col_half_continues_rate > 0.45:
                weight[1] += 1
            else:
                weight[2] += 1

            self._phase = max(weight, key=lambda x: weight[x])
        return self._phase

    def test(self):
        global COUNT
        ImageBase(255 - self.slice(Config.CELL_LEVEL_REGION)).save(str(COUNT) + ".tif", "../box_images/saved/level")
        ImageBase(255 - self.slice(Config.CELL_PHASE_REGION)).save(str(COUNT) + ".tif", "../box_images/saved/phase")
        COUNT += 1


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


def slope(data_list, key=lambda x: x):
    return (0 if i == 0 else int(key(data_list[i])) - int(key(data_list[i - 1])) for i, _ in enumerate(data_list))


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


if __name__ == '__main__':
    my_box = Box("../box_images/test")
    for i in my_box.images:
        for c in i.cells:
            print(c.name, c.phase, c.level)


