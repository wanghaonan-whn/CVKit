from cvkit.core.annotation.io.txt import TXTDocument


class YOLOSegmentationUtils(TXTDocument):

    def polygon_2_bbox(self):
        lines = self.readlines()
        if len(lines) == 0:
            raise ValueError(f"No polygon annotation in {self.path.name}")

        bbox = []
        for line in lines:
            values = line.strip().split()

            if len(values) < 7:
                raise ValueError("YOLO 多边形标签至少需要一个类别和三个坐标点")

            coordinate_count = len(values) - 1
            if coordinate_count % 2 != 0:
                raise ValueError(f"多边形坐标数量必须为偶数，当前为：{coordinate_count}")

            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
            x_coordinates = coordinates[0::2]
            y_coordinates = coordinates[1::2]

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

