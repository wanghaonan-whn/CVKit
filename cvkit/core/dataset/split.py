import random
import shutil
from pathlib import Path
from cvkit.core.dataset.base import Dataset
from concurrent.futures import ThreadPoolExecutor


class DatasetSplitter:
    def __init__(self, dataset: Dataset, ratio: float = 0.9) -> None:
        if not 0 < ratio < 1:
            raise ValueError("ratio must be between 0 and 1")

        self.dataset = dataset
        self.ratio = ratio

    def split(self, max_workers: int = 8) -> None:
        save_dir = self.dataset.data_dir / "split"

        if save_dir.exists():
            raise FileExistsError(f"Dir already exists: {save_dir}")

        img_train_dir = save_dir / "train" / "images"
        label_train_dir = save_dir / "train" / "labels"
        img_val_dir = save_dir / "val" / "images"
        label_val_dir = save_dir / "val" / "labels"

        for save_path in (img_train_dir, label_train_dir, img_val_dir, label_val_dir):
            save_path.mkdir(parents=True, exist_ok=True)

        labels = sorted(self.dataset.label_dir.glob("*.txt"))

        if not labels:
            raise FileNotFoundError(f"No label files found in {self.dataset.label_dir}")

        random.shuffle(labels)
        num_train = int(len(labels) * self.ratio)

        train_labels = labels[:num_train]
        val_labels = labels[num_train:]
        tasks = []

        for label_file in train_labels:
            tasks.append((label_file, img_train_dir, label_train_dir))

        for label_file in val_labels:
            tasks.append((label_file, img_val_dir, label_val_dir))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._copy_one, label_file, img_save_dir, label_save_dir)
                for label_file, img_save_dir, label_save_dir in tasks
            ]
            for future in futures:
                future.result()

    def _copy_one(self, label_file: Path, img_save_dir: Path, label_save_dir: Path) -> None:
        image_file = self.dataset.find_image(label_file.stem)
        if image_file is None:
            raise FileNotFoundError(f"Image not found: {label_file.stem}")

        shutil.copy2(label_file, label_save_dir / label_file.name)
        shutil.copy2(image_file, img_save_dir / image_file.name)
