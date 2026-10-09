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
    def calculate_iou(first: list[int | float], second: list[int | float]) -> float:
        _, x1, y1, width1, height1 = first
        _, x2, y2, width2, height2 = second

        first_xmin = x1 - width1 / 2
        first_ymin = y1 - height1 / 2
        first_xmax = x1 + width1 / 2
        first_ymax = y1 + height1 / 2

        second_xmin = x2 - width2 / 2
        second_ymin = y2 - height2 / 2
        second_xmax = x2 + width2 / 2
        second_ymax = y2 + height2 / 2

        intersection_width = max(0.0, min(first_xmax, second_xmax) - max(first_xmin, second_xmin))
        intersection_height = max(0.0, min(first_ymax, second_ymax) - max(first_ymin, second_ymin))

        intersection_area = intersection_width * intersection_height
        first_area = width1 * height1
        second_area = width2 * height2
        union_area = first_area + second_area - intersection_area

        if union_area <= 0:
            return 0.0
        return intersection_area / union_area

    @staticmethod
    def is_bbox_iou(first: list[int | float], second: list[int | float]) -> bool:
        iou = AnnotationUtils.calculate_iou(first, second)
        return iou > 0.0
