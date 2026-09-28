import cv2
from cvkit.core.annotation.hbb.yolo import YOLOAnnotationUtils
from cvkit.core.dataset.base import Dataset


class DatasetChecker:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def check(self) -> None:
        self.check_images()
        self.check_labels()

    def check_images(self) -> None:
        for image_file in self.dataset.image_dir.iterdir():
            if image_file.suffix.lower() not in self.dataset.IMAGE_EXTS:
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Bad image: {image_file}")

            label_file = self.dataset.label_dir / f"{image_file.stem}.txt"
            if not label_file.exists():
                print(f"Missing label: {image_file}")
                continue

            is_empty = not any(line.strip() for line in YOLOAnnotationUtils(label_file).readlines())
            if is_empty:
                print(f"Empty label: {label_file}")

    def check_labels(self) -> None:
        for label_file in self.dataset.label_dir.glob("*.txt"):
            image_file = self.dataset.find_image(label_file.stem)
            if image_file is None:
                print(f"Missing image: {label_file}")
