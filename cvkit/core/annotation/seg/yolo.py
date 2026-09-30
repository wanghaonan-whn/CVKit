from cvkit.core.annotation.annotation import YOLOAnnotationUtils
from cvkit.core.annotation.hbb.yolo import YOLODetectionUtils


class YOLOSegmentationUtils(YOLOAnnotationUtils):
    def to_detection(self):
        if not self.labels:
            raise ValueError(f"No polygon annotation in labels")

        bbox = []
        for class_id, *coordinates in self.labels:
            if len(coordinates) < 6 or len(coordinates) % 2 != 0:
                raise ValueError("Invalid YOLO segmentation annotation")
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
