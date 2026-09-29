import cv2
from cvkit.core.annotation.hbb.yolo import YOLODetectionUtils
from cvkit.core.dataset.base import Dataset
from cvkit.core.dataset.result import DatasetCheckResult


class DatasetChecker:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def check(self) -> DatasetCheckResult:
        result = DatasetCheckResult()
        self._check_images(result)
        self._check_labels(result)
        self._check_image_extensions(result)
        return result

    def _check_images(self, result: DatasetCheckResult) -> None:
        for image_file in self.dataset.image_dir.iterdir():
            if image_file.suffix.lower() != self.dataset.image_ext:
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                result.bad_images.append(image_file)
                continue

            label_file = self.dataset.label_dir / f"{image_file.stem}.txt"
            if not label_file.exists():
                result.missing_labels.append(image_file)
                continue

            is_empty = not any(line.strip() for line in YOLODetectionUtils(label_file).readlines())
            if is_empty:
                result.empty_labels.append(label_file)

    def _check_labels(self, result: DatasetCheckResult) -> None:
        for label_file in self.dataset.label_dir.glob("*.txt"):
            image_file = self.dataset.find_image(label_file.stem)
            if image_file is None:
                result.orphan_labels.append(label_file)

    def _check_image_extensions(self, result: DatasetCheckResult) -> None:
        for image_file in self.dataset.image_dir.iterdir():
            if not image_file.is_file():
                continue

            if image_file.suffix.lower() != self.dataset.image_ext:
                result.nonstandard_images.append(image_file)
