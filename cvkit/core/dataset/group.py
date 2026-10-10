from collections import defaultdict

from cvkit.core.dataset.base import Dataset
from cvkit.core.image.utils import ImageUtils


class DatasetGroupChecker:
    def __init__(self, datasets: list[Dataset]) -> None:
        self.datasets = datasets

    def find_repeat_images_group(self) -> list[list[str]]:
        hash_groups: dict[str, list[str]] = defaultdict(list)
        for dataset in self.datasets:
            for image_file in dataset.iter_images():
                image_hash = ImageUtils.calculate_file_hash(image_file)
                hash_groups[image_hash].append(str(image_file))
        return [image_files for image_files in hash_groups.values() if len(image_files) >= 2]
