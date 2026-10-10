import cv2
from pathlib import Path
from cvkit.core.annotation.hbb.yolo import YOLODetectionUtils
from cvkit.core.annotation.utils import AnnotationUtils
from cvkit.core.dataset.base import Dataset
from cvkit.core.dataset.result import DatasetCheckResult, DuplicateBBoxIssue
from cvkit.core.image.utils import ImageUtils


class DatasetChecker:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def check(self) -> DatasetCheckResult:
        result = DatasetCheckResult()
        self._check_images(result)
        self._check_orphan_images(result)
        self._check_empty_labels(result)
        self._check_orphan_labels(result)
        self._check_repeat_label(result)
        self._check_repeat_images(result)
        return result

    def _check_images(self, result: DatasetCheckResult) -> None:
        """ 检查损坏图片 """
        for image_file in self.dataset.iter_images():
            image = cv2.imread(str(image_file))
            if image is None:
                result.bad_images.append(image_file)

    def _check_orphan_images(self, result: DatasetCheckResult) -> None:
        for image_file in self.dataset.iter_images():
            label_file = self.dataset.label_dir / f"{image_file.stem}.txt"
            if not label_file.is_file():
                result.missing_labels.append(image_file)

    def _check_empty_labels(self, result: DatasetCheckResult) -> None:
        """ 检查空标签 """
        for label_file in self.dataset.iter_labels():
            try:
                annotation = YOLODetectionUtils.from_file(label_file)
            except (ValueError, OSError, UnicodeError) as exc:
                result.invalid_labels[label_file] = str(exc)
                continue

            if annotation.is_empty():
                result.empty_labels.append(label_file)

    def _check_orphan_labels(self, result: DatasetCheckResult) -> None:
        """ 检查孤儿标注 """
        image_index = self.dataset.build_image_index()
        result.ambiguous_images.update({stem: images for stem, images in image_index.items() if len(images) > 1})
        for label_file in self.dataset.iter_labels():
            if label_file.stem not in image_index:
                result.orphan_labels.append(label_file)

    def _check_repeat_label(self, result: DatasetCheckResult) -> None:
        for label_file in self.dataset.iter_labels():
            if label_file in result.invalid_labels:
                continue
            labels = YOLODetectionUtils.from_file(label_file).parse()
            for first_index, first_bbox in enumerate(labels):
                for second_index in range(first_index + 1, len(labels)):
                    second_bbox = labels[second_index]
                    iou = AnnotationUtils.calculate_iou(first_bbox, second_bbox)
                    if iou >= 0.95:
                        result.repeat_labels.append(
                            DuplicateBBoxIssue(
                                label_file=label_file,
                                first_index=first_index + 1,
                                second_index=second_index + 1,
                                iou=iou
                            )
                        )

    def _check_repeat_images(self, result: DatasetCheckResult) -> None:
        seen: dict[str, Path] = {}
        for image_file in self.dataset.iter_images():
            image_hash = ImageUtils.calculate_file_hash(image_file)
            if image_hash in seen:
                result.repeat_images.append((seen[image_hash], image_file))
            else:
                seen[image_hash] = image_file
