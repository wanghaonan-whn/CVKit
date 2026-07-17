import random
import shutil
from pathlib import Path

import cv2


class Datasets:
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    def __init__(self, data_dir: str | Path, ratio: float = 0.9) -> None:
        self.data_dir = Path(data_dir)
        self.ratio = ratio
        self.image_dir = Path(self.data_dir) / "images"
        if not 0 < self.ratio < 1:
            raise ValueError("ratio must be between 0 and 1")

    def check(self) -> "Datasets":
        """ 检测坏文件 """
        bad_dir = self.data_dir / "images_bad"
        image_list = self.image_dir.glob("*")
        image_list = [image for image in image_list if image.suffix in Datasets.IMAGE_EXTS]
        for image_file in image_list:
            image = cv2.imread(str(image_file))
            if image is None:
                save_image_path = bad_dir / image_file.name
                save_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(image_file, save_image_path)
                print(f"Moved: {image_file} -> {save_image_path}")
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
            for ext in Datasets.IMAGE_EXTS:
                img_file = self.image_dir / f"{stem}{ext}"
                if img_file.exists():
                    shutil.copy2(img_file, img_save_dir / img_file.name)
                    break
            else:
                raise FileNotFoundError(f"Image not found: {stem}")


if __name__ == "__main__":
    Datasets("/mnt/8T/TV/项点/速度传感器安装螺栓折断或丢失/datasets2").check()
