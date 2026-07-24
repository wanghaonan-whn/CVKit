from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import List

from cvkit.core.annotation.io.txt import TXTDocument


class YOLOAnnotationUtils(TXTDocument):
    """
        YOLO 标签工具类 V1.0
    """

    def parse_label(self) -> tuple[List[List[float]], List[int]]:
        bboxes = []
        classes = []
        for line in self.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls, x, y, w, h = parts
            bboxes.append([float(x), float(y), float(w), float(h)])
            classes.append(int(cls))
        return bboxes, classes

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

    def del_cls(self, src_cls: str | int) -> TXTDocument:
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
        readlines = self.readlines()
        if len(readlines) == 0:
            self.path.unlink()
        return self

    def remove_duplicate(self) -> YOLOAnnotationUtils:
        readlines = self.readlines()
        unique = list(dict.fromkeys(readlines))
        self.content = "".join(unique)
        self.save()
        return self

    def count_classes(self) -> Counter[int]:
        return Counter(
            int(line.split()[0])
            for line in self.readlines()
            if line.strip()
        )
