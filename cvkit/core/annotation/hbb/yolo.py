from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import List
from cvkit.core.annotation.io.txt import TxtDocument


class YOLOAnnotationUtils(TxtDocument):
    """
        YOLO 标签工具类 V1.0
    """

    @staticmethod
    def parse_label(lines: List) -> List[List[float | int]]:
        labels = []
        for line in lines:
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
        self.save()
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
        self.save()
        return self

    def remove_empty(self) -> YOLOAnnotationUtils:
        if len(self.readlines()) == 0:
            self.path.unlink()
        return self

    def remove_duplicate(self) -> YOLOAnnotationUtils:
        unique = list(dict.fromkeys(self.readlines()))
        self.content = "".join(unique)
        self.save()
        return self

    def count_classes(self) -> Counter[int]:
        return Counter(
            int(line.split()[0])
            for line in self.readlines()
            if line.strip()
        )

    def get_classes_box(self, class_ids: int | List[int]) -> List[str]:
        target_class_ids = {class_ids} if isinstance(class_ids, int) else set(class_ids)
        return [
            line
            for line in self.readlines()
            if line.strip() and int(line.split()[0]) in target_class_ids
        ]

    @staticmethod
    def yolo_to_xy(size, x, y, w, h):
        x = x * size[0]
        w = w * size[0]
        y = y * size[1]
        h = h * size[1]
        return x, y, w, h
