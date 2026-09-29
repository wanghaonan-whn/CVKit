from pathlib import Path
from typing import Mapping
from collections import Counter
from cvkit.core.annotation.io.txt import TxtDocument


class YOLOAnnotationUtils:
    """ YOLO 通用工具类 """

    def __init__(self, labels: list[list[int | float]]):
        self.labels = [[int(label[0]), *map(float, label[1:])] for label in labels]

    @classmethod
    def from_labels(cls, labels: list[list[int | float]]) -> "YOLOAnnotationUtils":
        return cls(labels)

    @classmethod
    def from_file(cls, path: Path | str) -> "YOLOAnnotationUtils":
        lines = TxtDocument(path).readlines()
        labels = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_, *coordinates = parts
            labels.append([int(class_), *map(float, coordinates)])
        return cls(labels)

    @property
    def length(self) -> int:
        return len(self.labels)

    def parse(self) -> list[list[float | int]]:
        """ 解析 """
        return [label.copy() for label in self.labels]

    def is_label_in_yolo(self, class_ids: list[str | int]) -> bool:
        """标签查找对应的yolo标签"""
        class_ids = set(map(int, class_ids))
        return any(label[0] in class_ids for label in self.labels)

    def is_empty(self) -> bool:
        return self.length == 0

    def count(self) -> Counter[int]:
        """ 计数 """
        return Counter(label[0] for label in self.labels)

    def remap(self, mapping: Mapping[int | str, int | str]) -> "YOLOAnnotationUtils":
        """ 类别映射，例如 {0: 1, 1: 0} """
        class_mapping = {int(k): int(v) for k, v in mapping.items()}
        for label in self.labels:
            label[0] = class_mapping.get(int(label[0]), int(label[0]))
        return self

    def retain(self, class_ids: list[str | int]) -> "YOLOAnnotationUtils":
        """ 保留 """
        class_ids = list(map(int, class_ids))
        self.labels = [label for label in self.labels if label[0] in class_ids]
        return self

    def delete_by_cls(self, class_ids: list[str | int]) -> "YOLOAnnotationUtils":
        """ 删除 """
        class_ids = list(map(int, class_ids))
        self.labels = [label for label in self.labels if label[0] not in class_ids]
        return self

    def remove_duplicate(self) -> "YOLOAnnotationUtils":
        """ 删重 """
        self.labels = [list(label) for label in dict.fromkeys(tuple(label) for label in self.labels)]
        return self

    def merge(self, other: "YOLOAnnotationUtils") -> "YOLOAnnotationUtils":
        self.labels.extend(label.copy() for label in other.labels)
        return self

    def get_classes_label(self, class_ids: int | list[str | int]) -> list[list[float | int]]:
        """ 获取指定cls """
        target_class_ids = {class_ids} if isinstance(class_ids, int) else set(list(map(int, class_ids)))
        return [label.copy() for label in self.labels if label[0] in target_class_ids]

    def save(self, path: str | Path) -> "YOLOAnnotationUtils":
        target = Path(path)
        TxtDocument.new(target).write("".join([" ".join(map(str, label)) + "\n" for label in self.labels])).save()
        return self
