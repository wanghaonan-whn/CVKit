from typing import List


class AnnotationUtils:
    @staticmethod
    def yolo_to_xywh(size, x, y, w, h):
        """ yolo转绝对坐标 """
        x = x * size[0]
        w = w * size[0]
        y = y * size[1]
        h = h * size[1]
        return x, y, w, h

    @staticmethod
    def yolo_to_voc(size, x, y, w, h):
        """ yolo转voc """
        center_x = x * size[0]
        center_y = y * size[1]
        w = w * size[0]
        h = h * size[1]

        xmin = center_x - w / 2
        ymin = center_y - h / 2
        xmax = center_x + w / 2
        ymax = center_y + h / 2
        return xmin, ymin, xmax, ymax

    @staticmethod
    def is_bbox_iou(first: List[int | float], second: List[int | float]) -> bool:
        _, x1, y1, w1, h1 = first
        _, x2, y2, w2, h2 = second

        first_xmin = x1 - w1 / 2
        first_ymin = y1 - h1 / 2
        first_xmax = x1 + w1 / 2
        first_ymax = y1 + h1 / 2

        second_xmin = x2 - w2 / 2
        second_ymin = y2 - h2 / 2
        second_xmax = x2 + w2 / 2
        second_ymax = y2 + h2 / 2

        intersection_width = min(first_xmax, second_xmax) - max(first_xmin, second_xmin)
        intersection_height = min(first_ymax, second_ymax) - max(first_ymin, second_ymin)
        return intersection_width > 0 and intersection_height > 0
