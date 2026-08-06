from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import List

from cvkit.core.annotation.hbb.voc import VOCAnnotationUtils
from cvkit.core.annotation.io.txt import TxtDocument


class YOLOAnnotationUtils(TxtDocument):
    """
        YOLO 标签工具类 V1.0
    """

    def parse_label(self) -> List[List[float | int]]:
        labels = []
        for line in self.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls, x, y, w, h = parts
            labels.append([int(cls), float(x), float(y), float(w), float(h)])
        return labels

    def remap_classes(self, mapping: Mapping[int | str, int | str]) -> YOLOAnnotationUtils:
        """
            Args:
                mapping: 类别映射，例如 {0: 1, 1: 0}。
            Returns:
                链式调用
            Examples:
                >>> YOLOAnnotationUtils("").remap_classes({0: 1})
                >>> YOLOAnnotationUtils("").remap_classes({0: 1, 1: 0})
         """
        mapping = {int(k): int(v) for k, v in mapping.items()}

        new_lines = []
        for line in self.readlines():
            parts = line.split()

            if not parts:
                new_lines.append(line)
                continue

            cls = int(parts[0])
            if cls in mapping:
                parts[0] = str(mapping[cls])
            new_lines.append(" ".join(parts) + "\n")

        self.content = "".join(new_lines)
        return self

    def del_cls(self, src_cls: str | int) -> YOLOAnnotationUtils:
        """ 删除类别 """
        src_cls = int(src_cls)

        new_lines = []
        for line in self.readlines():
            parts = line.split()

            if not parts:
                new_lines.append(line)
                continue

            if int(parts[0]) == src_cls:
                continue
            new_lines.append(line)
        self.content = "".join(new_lines)
        return self

    def remove_empty(self) -> YOLOAnnotationUtils:
        if len(self.readlines()) == 0:
            self.path.unlink()
        return self

    def remove_duplicate(self) -> YOLOAnnotationUtils:
        unique = list(dict.fromkeys(self.readlines()))
        self.content = "".join(unique)
        return self

    def count_classes(self) -> Counter[int]:
        return Counter(
            int(line.split()[0])
            for line in self.readlines()
            if line.strip()
        )

    def get_classes_box(self, class_ids: int | List[int]) -> List[List[float | int]]:
        target_class_ids = {class_ids} if isinstance(class_ids, int) else set(class_ids)
        return [
            line
            for line in self.parse_label()
            if line[0] in target_class_ids
        ]

    @staticmethod
    def yolo_to_xywh(size, x, y, w, h):
        x = x * size[0]
        w = w * size[0]
        y = y * size[1]
        h = h * size[1]
        return x, y, w, h

    @staticmethod
    def yolo_to_voc(size, x, y, w, h):
        center_x = x * size[0]
        center_y = y * size[1]
        w = w * size[0]
        h = h * size[1]

        xmin = center_x - w / 2
        ymin = center_y - h / 2
        xmax = center_x + w / 2
        ymax = center_y + h / 2
        return xmin, ymin, xmax, ymax

    def save_as_voc(
            self,
            img_name: str,
            img_size: tuple[int, int],
            classes_mapping: Mapping[int, str],
            save_dir: str | Path | None = None,
            depth: int = 1
    ) -> YOLOAnnotationUtils:
        if save_dir is None:
            save_dir = self.path.parents[1].joinpath("xml")
        else:
            save_dir = Path(save_dir)

        save_path = save_dir / f"{self.path.stem}.xml"
        document = VOCAnnotationUtils.build(
            img_name=img_name,
            img_size=img_size,
            bboxes=[],
            save_path=save_path,
            depth=depth,
        )
        unknown_class_ids = set(self.count_classes().keys()) - set(classes_mapping)
        if unknown_class_ids:
            raise ValueError(f"Classes missing from classes_mapping: {sorted(unknown_class_ids)}")

        labels = self.parse_label()
        for class_id, x, y, box_width, box_height in labels:
            xmin, ymin, xmax, ymax = self.yolo_to_voc(img_size, x, y, box_width, box_height)
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            document.append_object(class_name=classes_mapping[int(class_id)], bbox=bbox)

        document.save()
        return self


if __name__ == "__main__":
    txt_path = "/mnt/FourT/test/labels/1.txt"
    YOLOAnnotationUtils(txt_path).save_as_voc(
        img_name="test",
        img_size=(7644, 430),
        classes_mapping={0: "abc"},
    )
