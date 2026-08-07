import random
import shutil
import cv2
from pathlib import Path
from cvkit.core.annotation.hbb.yolo import YOLOAnnotationUtils


class Datasets:
    # TODO: 数据集检查会直接修改数据，建议职责区分。DatasetsCleaner
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self, data_dir: str | Path, ratio: float = 0.9) -> None:
        self.data_dir = Path(data_dir)
        self.ratio = ratio
        self.image_dir = Path(self.data_dir) / "images"
        if not 0 < self.ratio < 1:
            raise ValueError("ratio must be between 0 and 1")

    def check(self) -> "Datasets":
        """
            1. 检测坏文件
            2. 检查空标签
            3. 检查孤儿label
        """
        bad_dir = self.data_dir / "images_bad"
        image_list = self.image_dir.glob("*")
        image_list = [image for image in image_list if image.suffix.lower() in Datasets.IMAGE_EXTS]
        for image_file in image_list:
            image = cv2.imread(str(image_file))
            if image is None:
                save_image_path = bad_dir / image_file.name
                save_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(image_file, save_image_path)
                print(f"Moved: {image_file} -> {save_image_path}")
            label_path = self.data_dir / "labels" / f"{image_file.stem}.txt"
            if not label_path.exists():
                print(f"Missing label: {image_file} -> {label_path}")
                continue
            is_empty = not any(line.strip() for line in YOLOAnnotationUtils(label_path).readlines())
            if is_empty:
                print(f"Empty: {image_file} -> {label_path}")

        label_dir = self.data_dir / "labels"
        for label_file in label_dir.glob("*.txt"):
            image_file = self.__find_image(label_file.stem)
            if image_file is None:
                print(f"Missing image: {label_file}")
        return self

    def split(self) -> "Datasets":
        label_dir = Path(self.data_dir) / "labels"
        save_dir = Path(self.data_dir) / "split"

        img_train_dir = save_dir / "train" / "images"
        label_train_dir = save_dir / "train" / "labels"
        img_val_dir = save_dir / "val" / "images"
        label_val_dir = save_dir / "val" / "labels"

        for save_path in (img_train_dir, label_train_dir, img_val_dir, label_val_dir):
            save_path.mkdir(parents=True, exist_ok=True)

        labels = sorted(label_dir.glob("*.txt"))
        if not labels:
            raise FileNotFoundError(f"No label files found in {label_dir}")

        random.shuffle(labels)

        num_train = int(len(labels) * self.ratio)
        train_labels = labels[:num_train]
        val_labels = labels[num_train:]

        self.__copy_dataset(train_labels, img_train_dir, label_train_dir)
        self.__copy_dataset(val_labels, img_val_dir, label_val_dir)
        return self

    def __copy_dataset(self, labels: list[Path], img_save_dir: Path, label_save_dir: Path) -> None:
        for label_file in labels:
            shutil.copy2(label_file, label_save_dir / label_file.name)
            stem = label_file.stem
            image_file = next(
                (
                    image_file
                    for image_file in self.image_dir.glob(f"{stem}.*")
                    if image_file.suffix.lower() in self.IMAGE_EXTS
                ),
                None,
            )
            if image_file is None:
                raise FileNotFoundError(f"Image not found: {stem}")
            shutil.copy2(image_file, img_save_dir / image_file.name)

    def __find_image(self, stem: str) -> Path | None:
        return next(
            (
                image_file
                for image_file in self.image_dir.glob(f"{stem}.*")
                if image_file.suffix.lower() in self.IMAGE_EXTS
            ),
            None,
        )


if __name__ == "__main__":
    Datasets("/mnt/FourT/TV/项点/定位/纵向牵引拉杆/datasets2", ratio=0.5).split()
