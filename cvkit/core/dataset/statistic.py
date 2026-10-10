from collections import Counter
from cvkit.core.dataset.base import Dataset
from cvkit.core.annotation.hbb.yolo import YOLODetectionUtils


class DatasetStatistics:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def count_images(self) -> int:
        return sum(1 for _ in self.dataset.iter_images())

    def count_labels(self) -> int:
        return sum(1 for _ in self.dataset.iter_labels())

    def count_classes(self) -> Counter[int]:
        counter = Counter()
        for label in self.dataset.iter_labels():
            counter.update(YOLODetectionUtils.from_file(label).count())
        return counter

    def images_per_class(self) -> Counter[int]:
        counter = Counter()
        for label_file in self.dataset.iter_labels():
            counter.update(YOLODetectionUtils.from_file(label_file).count().keys())
        return counter
