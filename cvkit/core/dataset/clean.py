import shutil
import cv2
from cvkit.core.dataset.base import Dataset


class DatasetCleaner:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def move_bad_images(self) -> None:
        bad_dir = self.dataset.data_dir / "images_bad"
        bad_dir.mkdir(parents=True, exist_ok=True)

        for image_file in self.dataset.image_dir.iterdir():
            if image_file.suffix.lower() not in self.dataset.IMAGE_EXTS:
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                save_path = bad_dir / image_file.name
                shutil.move(image_file, save_path)
                print(f"Moved: {image_file} -> {save_path}")