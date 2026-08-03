from typing import List

from cvkit.core.annotation.io.txt import TxtDocument


class YOLOSegmentationUtils(TxtDocument):

    @staticmethod
    def parse_label(lines: List[str]) -> list[tuple[int, list[tuple[float, float]]]]:
        """
            :return: [
                      (0, [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4)]),
                      (1, [(0.5, 0.5), (0.6, 0.5), (0.6, 0.7)]),
                    ]
        """
        labels = []
        for line_number, line in enumerate(lines, start=1):
            values = line.strip().split()
            if not values: continue
            if len(values) < 7:
                raise ValueError(f"Line {line_number}: at least three points are required")
            if (len(values) - 1) % 2 != 0:
                raise ValueError(f"Line {line_number}: coordinate count must be even")

            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
            points = list(zip(coordinates[0::2], coordinates[1::2]))
            labels.append((class_id, points))
        return labels

    def polygon_2_bbox(self):
        annotations = self.parse_label(self.readlines())
        if not annotations:
            raise ValueError(f"No polygon annotation in {self.path.name}")

        bbox = []
        for class_id, points in annotations:
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
            bbox.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        self.content = "\n".join(bbox) + "\n"
        self.write(self.content)
        return self
