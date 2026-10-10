import random
import shutil
from pathlib import Path
from cvkit.core.dataset.base import Dataset
from concurrent.futures import ThreadPoolExecutor


class DatasetSplitter:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def split(self, ratio: float = 0.9, seed: int | None = 42, max_workers: int = 8) -> None:
        if not 0 < ratio < 1:
            raise ValueError("ratio must be between 0 and 1")

        save_dir = self.dataset.data_dir / "split"
        if save_dir.exists():
            raise FileExistsError(f"Dir already exists: {save_dir}")

        labels = sorted(self.dataset.iter_labels())
        if not labels:
            raise FileNotFoundError(f"No label files found in {self.dataset.label_dir}")
        image_index = self.dataset.build_image_index()
        pairs = []

        for label_file in labels:
            image_file = self.dataset.find_image(label_file.stem, image_index)
            if image_file is None:
                raise FileNotFoundError(f"Image not found: {label_file.stem}")
            pairs.append((image_file, label_file))

        rng = random.Random(seed)
        rng.shuffle(pairs)
        num_train = int(len(pairs) * ratio)
        img_train_dir = save_dir / "train" / "images"
        label_train_dir = save_dir / "train" / "labels"
        img_val_dir = save_dir / "val" / "images"
        label_val_dir = save_dir / "val" / "labels"
        for save_path in (img_train_dir, label_train_dir, img_val_dir, label_val_dir):
            save_path.mkdir(parents=True, exist_ok=True)

        tasks = [(image, label, img_train_dir, label_train_dir) for image, label in pairs[:num_train]]
        tasks.extend((image, label, img_val_dir, label_val_dir) for image, label in pairs[num_train:])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._copy_one, *task) for task in tasks]
            for future in futures:
                future.result()

    @staticmethod
    def _copy_one(image_file: Path, label_file: Path, img_save_dir: Path, label_save_dir: Path) -> None:
        shutil.copy2(image_file, img_save_dir / image_file.name)
        shutil.copy2(label_file, label_save_dir / label_file.name)
