from cvkit.core.annotation.annotation import YOLOAnnotationUtils
from cvkit.core.annotation.hbb.yolo import YOLODetectionUtils


class YOLOSegmentationUtils(YOLOAnnotationUtils):
    def validate(self):
        for label in self.labels:
            class_id, *coordinates = label
            if len(coordinates) < 6:
                raise ValueError(f"Segmentation label must contain at least 3 points: {label}")
            if len(coordinates) % 2 != 0:
                raise ValueError(f"Segmentation label has an odd number of coordinates: {label}")
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                raise ValueError(f"Segmentation label contains coordinates outside [0, 1]: {label}")

            points = list(zip(coordinates[::2], coordinates[1::2]))
            if len(set(points)) < 3:
                raise ValueError(f"Segmentation label has fewer than 3 distinct points: {label}")

            double_area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points,points[1:] + points[:1]))
            if abs(double_area) <= 1e-12:
                raise ValueError(f"Segmentation label has zero polygon area: {label}")

    def to_detection(self):
        bbox = []
        for class_id, *coordinates in self.labels:
            points = list(zip(coordinates[0::2], coordinates[1::2]))
            x_coordinates = [x for x, _ in points]
            y_coordinates = [y for _, y in points]

            x_min = min(x_coordinates)
            y_min = min(y_coordinates)
            x_max = max(x_coordinates)
            y_max = max(y_coordinates)

            box_width = x_max - x_min
            box_height = y_max - y_min

            if box_width <= 0 or box_height <= 0:
                raise ValueError("多边形无法生成有效矩形框：", f"width={box_width}, height={box_height}")

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            bbox.append([int(class_id), x_center, y_center, box_width, box_height])
        return YOLODetectionUtils(bbox)
