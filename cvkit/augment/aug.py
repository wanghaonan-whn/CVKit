import albumentations as A
import cv2
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from cvkit.core.annotation.hbb.yolo import YOLOAnnotationUtils
from cvkit.core.annotation.io.txt import TxtDocument


class Augmenter:
    """
        基于 Albumentations 数据增强构建器
        Example:
            >>> transform = (
            ...     Augmenter()
            ...     .horizontal_flip()
            ...     .rotate(limit=10)
            ...     .brightness()
            ...     .build()
            ... )
    """

    def __init__(self, path: str | Path, repeat: int = 10):
        self.path = Path(path)
        self.repeat = repeat
        self.__transforms = []
        self.save_label_dir = None
        self.save_image_dir = None

    def build(self, bbox=False):
        if bbox:
            return A.Compose(self.__transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]))
        return A.Compose(self.__transforms)

    def horizontal_flip(self, p=0.5) -> "Augmenter":
        """ 左右翻转 """
        self.__transforms.append(A.HorizontalFlip(p=p))
        return self

    def vertical_flip(self, p=0.5) -> "Augmenter":
        """ 上下翻转 """
        self.__transforms.append(A.VerticalFlip(p=p))
        return self

    def rotate(self, limit=15, p=0.5) -> "Augmenter":
        self.__transforms.append(A.Rotate(limit=limit, p=p))
        return self

    def brightness(self, brightness_limit=0.2, contrast_limit=0.2, p=0.5) -> "Augmenter":
        """
            亮度对比度
            Args:
                brightness_limit:
                    亮度变化范围，0.2 表示亮度随机变化 ±20%。
                contrast_limit:
                    对比度变化范围，0.2 表示对比度随机变化 ±20%。
                p:
                    执行该增强的概率。
            Returns:
                当前 Augmenter 对象。
        """
        self.__transforms.append(
            A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=p)
        )
        return self

    def blur(self, blur_limit=3, p=0.3) -> "Augmenter":
        """
            随机模糊。
            Args:
                blur_limit:
                    模糊核大小
                    - int，例如 ``3``。
                    - tuple，例如 ``(3, 7)``。
        """
        self.__transforms.append(A.Blur(blur_limit=blur_limit, p=p))
        return self

    def gauss_noise(self, std_range=(0.03, 0.08), p=0.5) -> "Augmenter":
        self.__transforms.append(A.GaussNoise(std_range=std_range, p=p))
        return self

    def image_compression(self, quality_range=(40, 100), p=0.8) -> "Augmenter":
        self.__transforms.append(A.ImageCompression(quality_range=quality_range, p=p))
        return self

    def clahe(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.5):
        self.__transforms.append(
            A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, p=p)
        )
        return self

    def to_gray(self) -> "Augmenter":
        self.__transforms.append(A.ToGray(p=0.2))
        return self

    def motion_blur(self, blur_limit=5, p=0.3) -> "Augmenter":
        self.__transforms.append(A.MotionBlur(blur_limit=blur_limit, p=p))
        return self

    def one_of_sharp_blur(self, p=0.4) -> "Augmenter":
        self.__transforms.append(
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.GaussianBlur(blur_limit=5, p=1),
                ],
                p=p,
            )
        )
        return self

    def one_of_affine(
            self,
            x_scale: tuple[float, float] = (0.8, 1.2),
            y_scale: tuple[float, float] = (0.9, 1.1),
            p=0.5,
            sub_p=1,
    ) -> "Augmenter":
        """ 拉伸组合增强 """
        self.__transforms.append(
            A.OneOf([
                A.Affine(scale={"x": x_scale}, interpolation=cv2.INTER_AREA, fit_output=True, p=sub_p),
                A.Affine(scale={"y": y_scale}, interpolation=cv2.INTER_AREA, fit_output=True, p=sub_p),
            ], p=p)
        )
        return self

    def augment(self, worker: int = 1) -> "Augmenter":
        image_dir = self.path / "images"
        save_dir = self.path / "aug"

        self.save_image_dir = save_dir / "images"
        self.save_label_dir = save_dir / "labels"
        self.save_image_dir.mkdir(parents=True, exist_ok=True)
        self.save_label_dir.mkdir(parents=True, exist_ok=True)

        image_paths = image_dir.iterdir()
        if worker <= 1:
            for image_path in image_paths:
                self._mp_augment(image_path)
        else:
            with ProcessPoolExecutor(max_workers=worker) as executor:
                list(executor.map(self._mp_augment, image_paths))

        return self

    def _mp_augment(self, image_path) -> None:
        image = cv2.imread(str(image_path))
        if image is None: return

        label_path = self.path / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"warning: {label_path} does not exist")
            return

        lines = YOLOAnnotationUtils(label_path).readlines()
        labels = [label[0] for label in YOLOAnnotationUtils.parse_label(lines)]
        bboxes = [label[1:] for label in YOLOAnnotationUtils.parse_label(lines)]
        transform = self.build(bbox=True)

        for index in range(self.repeat):
            augmented = transform(image=image, bboxes=bboxes, labels=labels)
            save_stem = f"{image_path.stem}-aug{index}"
            save_image_path = self.save_image_dir / f"{save_stem}{image_path.suffix}"
            save_label_path = self.save_label_dir / f"{save_stem}.txt"

            cv2.imwrite(str(save_image_path), augmented["image"])
            content = "".join(
                f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                for (x, y, w, h), cls in zip(augmented["bboxes"], augmented["labels"])
            )
            TxtDocument(save_label_path).write(content).save()
