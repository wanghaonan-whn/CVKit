from collections import Counter
from cvkit.core.dataset.base import Dataset
from cvkit.core.annotation.hbb.yolo import YOLODetectionUtils


class DatasetStatistics:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def count_images(self) -> int:
        return sum(1 for _ in self.dataset.image_dir.glob(f"*{self.dataset.image_ext}"))

    def count_labels(self) -> int:
        return sum(1 for _ in self.dataset.label_dir.glob("*.txt"))

    def count_classes(self) -> Counter[int]:
        counter = Counter()
        for label in self.dataset.label_dir.glob("*.txt"):
            counter += YOLODetectionUtils(label).count()
        return counter

    def images_per_class(self) -> Counter[int]:
        counter = Counter()
        for label in self.dataset.label_dir.glob("*.txt"):
            for key in YOLODetectionUtils(label).count().keys():
                counter[key] += 1
        return counter
