import os
import numpy as np

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


class Config:
    DEBUG = True

    ROW_SLOPE_THRESHOLD = 140000
    COLUMN_SLOPE_THRESHOLD = 40000  # first line
    LINES_MERGE_BLUR = 2
    CONTINUAL_LINE_THRESHOLD = 20
    BOTTOM_ROW_SLOPE_THRESHOLD = 1024


class Box:
    """
        For a group of images.
    """
    def __init__(self, images_path):
        self.grid_width_gap = None
        self.grid_width_raw = None
        self.grid_width = None
        self.grid_height = None
        self.in_cell_text_region = None  # ((0, 0), (10, 10))
        self.in_cell_level_region = None  # ((0, 0), (10, 10))
        self.in_cell_elite_region = None  # ((0, 0), (10, 10))
        self.grid_y_starts = None

        self.images = []
        for path, _, files in os.walk(images_path):
            for f in files:
                self.images.append(BoxImage(self, os.path.join(path, f)))

        self.employees = []


class BoxImage:
    """
    For one image.
    """
    def __init__(self, box, file_path):
        self.box = box

        self.cells = []
        self.grid_lines = []
        self.grid_x_starts = None

        self.img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # self.img_test = cv2.blur(self.img_gray, (2, 1))
        # for griding
        _, self.img_binary_220 = cv2.threshold(self.img_gray, 220, 255, cv2.THRESH_BINARY)
        self.img = self.img_binary_220
        # for identifying
        _, self.img_binary_168 = cv2.threshold(self.img_gray, 168, 255, cv2.THRESH_BINARY)

        self.split_lines()
        self.split_cells_in_line()

    def split_lines(self):
        if not self.box.grid_y_starts or not self.box.grid_height:
            row_sum = self.img_binary_220.sum(axis=1)
            row_length = len(row_sum)
            row_half_slope = [0 if i == 0 or i == row_length - 1 else int(row_sum[i + 1]) - int(row_sum[i - 1]) for i in
                              range(row_length)[::2]]

            # mid_index = row_sum.argmax()
            row_edges = []
            for index, value in enumerate(row_half_slope):
                if abs(value) > Config.ROW_SLOPE_THRESHOLD:
                    # ignore close lines.
                    if row_edges and row_edges[-1][1] ^ value >= 0 and index - row_edges[-1][0] <= Config.LINES_MERGE_BLUR:
                        continue
                    mark(self.img_binary_220[index * 2])
                    row_edges.append((index*2, value))
            show(self.img_binary_220)

            # count distance
            row_distances = [r[0] - row_edges[index - 1][0] for index, r in enumerate(row_edges) if index]
            self.box.grid_height = max(row_distances)

            # find grid_y_starts
            self.box.grid_y_starts = []
            for e in row_edges:
                if e[1] < 0:
                    self.box.grid_y_starts.append(e[0])
            if not len(self.box.grid_y_starts) == 2:
                self.box.grid_y_starts = None
                raise Exception("Error finding grid_y_starts:", self.box.grid_y_starts)

        self.grid_lines = list(self.img_binary_220[y_start:y_start + self.box.grid_height, :] for y_start in self.box.grid_y_starts)

    def split_cells_in_line(self):
        for line in self.grid_lines:
            if not self.box.grid_width or not self.grid_x_starts:
                column_sum = line.sum(axis=0)
                column_length = len(column_sum)
                column_half_slope = [0 if i == 0 or i == column_length - 1 else int(column_sum[i + 1]) - int(column_sum[i - 1])
                                     for i in range(column_length)]
                # mid_index = row_sum.argmax()
                column_edges = []
                for index, value in enumerate(column_half_slope):
                    if abs(value) > Config.COLUMN_SLOPE_THRESHOLD:
                        if column_edges and column_edges[-1][1] ^ value >= 0 and index - column_edges[-1][0] <= Config.LINES_MERGE_BLUR:
                            continue
                        column_edges.append((index, value))

                if not self.box.grid_width:
                    column_distances = [c[0] - column_edges[index - 1][0] for index, c in enumerate(column_edges) if
                                        index]
                    result = find_most_confident(column_distances)[:2]
                    result.sort(key=lambda x: x[0])
                    self.box.grid_width_gap = int(result[0][0])
                    self.box.grid_width_raw = int(result[1][0])
                    self.box.grid_width = int(result[1][0] + result[0][0] / 4)

                if not self.grid_x_starts:
                    # find seed
                    seed = None
                    for index, (e, v) in enumerate(column_edges):
                        if not index:
                            continue
                        if v < 0 and e - column_edges[index - 1][0] == self.box.grid_width_gap:
                            seed = e
                            break
                    if not seed:
                        raise Exception("Error finding x_seed")

                    # find grid_x_starts
                    distance = self.box.grid_width_raw + self.box.grid_width_gap
                    self.grid_x_starts = [int(seed + x*distance) for x in range(
                        -int(seed/distance), int((column_length-seed)/distance))]
                    # add last
                    if self.grid_x_starts[-1] + distance + self.box.grid_width_raw < column_length:
                        self.grid_x_starts.append(self.grid_x_starts[-1] + distance)

            for x_start in self.grid_x_starts:
                self.cells.append(Cell(self, line[:, x_start:x_start + self.box.grid_width]))

    def split_sections(self):
        if not self.box.in_cell_text_region:
            text_ranges = list(cell.find_text() for cell in self.cells)
            y_starts = [y[0] for y in text_ranges if y]
            y_ends = [y[1] for y in text_ranges if y]
            y_start = find_most_confident(y_starts, choose_func=min)[0][0]
            y_end = find_most_confident(y_ends, choose_func=max)[0][0]
            self.box.in_cell_text_region = (y_start, y_end)
        if not self.box.in_cell_level_region:
            pass


class Cell:
    """
    For one employee.
    """
    def __init__(self, box_image, img):
        self.box_image = box_image
        self.img = img

    def find_text(self):
        bottom_flag = True
        y_end = len(self.img)
        for index, row in enumerate(self.img[::-1]):
            row_slope_sum = sum(0 if j == 0 else abs(int(value) - int(row[j-1])) for j, value in enumerate(row))
            # print(row_slope_sum)
            if bottom_flag and row_slope_sum > Config.BOTTOM_ROW_SLOPE_THRESHOLD:
                y_end = len(self.img) - index
                mark(self.img[y_end])
                bottom_flag = False
            elif not bottom_flag and row_slope_sum < Config.BOTTOM_ROW_SLOPE_THRESHOLD/4:
                y_start = len(self.img) - index
                mark(self.img[y_start])
                break
        else:
            return None
        show(self)
        return y_start, y_end

    def read_info(self):
        if not (self.box_image.box.in_cell_text_region and self.box_image.box.in_cell_level_region and
                self.box_image.box.in_cell_elite_region):
            self.box_image.split_sections()
        text_region = self.box_image.box.in_cell_text_region
        show(255 - self.img[text_region[0]:text_region[1]])

        print(pytesseract.image_to_string(255 - self.img[text_region[0]:text_region[1]]))


def find_most_confident(data_list,
                        blur_func=lambda x, y: abs(x-y) <= Config.LINES_MERGE_BLUR,
                        choose_func=lambda x: round(np.average(x))):
    """
    find most typical values in list
    :param data_list:
    :param blur_func: return True to see similar value as one.
    :param choose_func: define which one to choose in all similar values.
    :return: value, confidence
    """
    # statistic
    values = {}
    for data in data_list:
        if values.get(data):
            values[data] += 1
        else:
            values[data] = 1
    # sort
    results = list(values.items())
    results.sort(key=lambda x: x[0])
    merged_groups = []
    for i in range(len(results)):
        if not i or not blur_func(results[i][0], results[i-1][0]):
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
    def wrapper(*args, **kwargs):
        if Config.DEBUG:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


@debug
def show(img):
    try:
        cv2.imshow('test', img)
        cv2.waitKey(0)
    except:
        show(img.img)
        # cv2.waitKey(0)


@debug
def mark(img, color=128):
    if type(img[0]) != np.ndarray:
        img.fill(color)
    else:
        for i in img:
            mark(i, color)


if __name__ == '__main__':
    my_box = Box("../box_images/test")
    my_box.images[0].cells[0].read_info()
    # cv2.destroyAllWindows()
