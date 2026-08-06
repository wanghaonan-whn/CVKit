import cv2
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cvkit.augment.aug import Augmenter
from cvkit.core.annotation.hbb.yolo import YOLOAnnotationUtils
from cvkit.core.annotation.io.txt import TxtDocument


class YOLODatasetAugmenter:
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, path, transform, repeat: int = 10):
        if repeat <= 0:
            raise ValueError("repeat must be greater than 0")

        self.path = Path(path)
        self.transform = transform
        self.repeat = repeat
        self.save_image_dir = self.path / "aug" / "images"
        self.save_label_dir = self.path / "aug" / "labels"

    def run(self, worker: int = 1) -> "YOLODatasetAugmenter":
        self.save_image_dir.mkdir(parents=True, exist_ok=True)
        self.save_label_dir.mkdir(parents=True, exist_ok=True)

        image_dir = self.path / "images"
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTS
        )

        if worker <= 1:
            for image_path in image_paths:
                self._augment_one(image_path)
        else:
            with ProcessPoolExecutor(max_workers=worker) as executor:
                list(executor.map(self._augment_one, image_paths))

        return self

    def _augment_one(self, image_path: Path) -> None:
        image = cv2.imread(str(image_path))
        if image is None:
            return

        label_path = self.path / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"warning: {label_path} does not exist")
            return

        parsed_labels = YOLOAnnotationUtils(label_path).parse_label()

        classes = [label[0] for label in parsed_labels]
        bboxes = [label[1:] for label in parsed_labels]

        for index in range(self.repeat):
            result = self.transform(image=image, bboxes=bboxes, labels=classes)

            save_stem = f"{image_path.stem}-aug{index}"
            save_image_path = self.save_image_dir / f"{save_stem}{image_path.suffix}"
            save_label_path = self.save_label_dir / f"{save_stem}.txt"

            if not cv2.imwrite(str(save_image_path), result["image"]):
                raise IOError(f"Failed to save image: {save_image_path}")

            content = "".join(
                f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                for (x, y, w, h), cls
                in zip(result["bboxes"], result["labels"])
            )
            TxtDocument.new(save_label_path).write(content).save()


if __name__ == "__main__":
    pass
    # path = ""
    # transform = (
    #     Augmenter()
    #     .gauss_noise(p=0.5)
    #     .one_of_affine()
    #     .to_gray(p=1)
    #     .build(bbox=True)
    # )
    # YOLODatasetAugmenter(
    #     path,
    #     transform=transform
    # ).run(worker=10)
