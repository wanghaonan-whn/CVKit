from cvkit.core.annotation.annotation import YOLOAnnotationUtils


class YOLOSegmentationUtils(YOLOAnnotationUtils):
    def polygon_2_bbox(self):
        annotations = self.parse()
        if not annotations:
            raise ValueError(f"No polygon annotation in {self.path.name}")

        bbox = []
        for class_id, *points in annotations:
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
