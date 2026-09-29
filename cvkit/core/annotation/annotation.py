from pathlib import Path
from typing import List, Mapping
from collections import Counter

from cvkit.core.annotation.io.txt import TxtDocument
from cvkit.core.annotation.utils import AnnotationUtils


class YOLOAnnotationUtils(TxtDocument):
    """ YOLO 通用工具类 """

    @property
    def length(self):
        return len(self.readlines())

    def parse(self) -> List[List[float | int]]:
        """ 解析 """
        labels = []
        for line in self.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls, *coordinates = parts
            labels.append([int(cls), *map(float, coordinates)])
        return labels

    def is_label_in_yolo(self, class_ids: List[str | int]) -> bool:
        """标签查找对应的yolo标签"""
        class_ids = set(map(int, class_ids))
        return all(int(label.split(' ')[0]) in class_ids for label in self.readlines())

    def count(self) -> Counter[int]:
        """ 计数 """
        return Counter(int(line.split()[0]) for line in self.readlines() if line.strip())

    def remap(self, mapping: Mapping[int | str, int | str]) -> "YOLOAnnotationUtils":
        """
            Args:
                mapping: 类别映射，例如 {0: 1, 1: 0}。
            Returns:
                链式调用
            Examples:
                >>> YOLOAnnotationUtils("").remap({0: 1})
                >>> YOLOAnnotationUtils("").remap({0: 1, 1: 0})
         """
        class_mapping = {int(k): int(v) for k, v in mapping.items()}

        new_lines = []
        for label in self.parse():
            class_id = int(label[0])
            label[0] = class_mapping.get(class_id, class_id)
            new_lines.append(" ".join(map(str, label)) + "\n")

        self.content = "".join(new_lines)
        return self

    def retain(self, retain: List[str | int]) -> "YOLOAnnotationUtils":
        """ 保留 """
        retain = list(map(int, retain))

        new_lines = []
        for label in self.parse():
            if label[0] not in retain:
                continue
            new_lines.append(" ".join(map(str, label)) + "\n")

        self.content = "".join(new_lines)
        return self

    def del_cls(self, delete: List[str | int]) -> "YOLOAnnotationUtils":
        """ 删除 """
        delete = list(map(int, delete))

        new_lines = []
        for label in self.parse():
            if label[0] in delete:
                continue
            new_lines.append(" ".join(map(str, label)) + "\n")

        self.content = "".join(new_lines)
        return self

    def remove_empty(self) -> "YOLOAnnotationUtils":
        """ 删空 """
        if len(self.readlines()) == 0:
            self.path.unlink()
        return self

    def remove_duplicate(self) -> "YOLOAnnotationUtils":
        """ 删重 """
        unique = list(dict.fromkeys(self.readlines()))
        self.content = "".join(unique)
        return self

    def merge(self, other: "str | Path | YOLOAnnotationUtils") -> "YOLOAnnotationUtils":
        if isinstance(other, YOLOAnnotationUtils):
            other_document = other
        else:
            other_document = YOLOAnnotationUtils(other)

        labels = self.parse() + other_document.parse()
        self.content = "".join(" ".join(map(str, label)) + "\n" for label in labels)
        return self

    def get_classes_label(self, class_ids: int | List[str | int]) -> List[List[float | int]]:
        """ 获取指定cls """
        target_class_ids = {class_ids} if isinstance(class_ids, int) else set(list(map(int, class_ids)))
        return [line for line in self.parse() if line[0] in target_class_ids]
