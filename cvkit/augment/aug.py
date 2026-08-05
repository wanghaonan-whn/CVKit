import albumentations as A
import cv2


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

    def __init__(self):
        self.__transforms = []

    def build(self, bbox=False):
        if bbox:
            return A.Compose(self.__transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]))
        return A.Compose(self.__transforms)

    def horizontal_flip(self, p=0.25) -> "Augmenter":
        """ 左右翻转 """
        self.__transforms.append(A.HorizontalFlip(p=p))
        return self

    def vertical_flip(self, p=0.25) -> "Augmenter":
        """ 上下翻转 """
        self.__transforms.append(A.VerticalFlip(p=p))
        return self

    def rotate(self, limit=15, p=0.25) -> "Augmenter":
        self.__transforms.append(A.Rotate(limit=limit, p=p))
        return self

    def brightness(self, brightness_limit=0.3, contrast_limit=0.1, p=0.25) -> "Augmenter":
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

    def blur(self, blur_limit=3, p=0.25) -> "Augmenter":
        self.__transforms.append(A.Blur(blur_limit=blur_limit, p=p))
        return self

    def gauss_noise(self, std_range=(0.0088,0.0152), p=0.25) -> "Augmenter":
        self.__transforms.append(A.GaussNoise(std_range=std_range, p=p))
        return self

    def image_compression(self, quality_range=(40, 100), p=0.25) -> "Augmenter":
        self.__transforms.append(A.ImageCompression(quality_range=quality_range, p=p))
        return self

    def clahe(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.25):
        self.__transforms.append(
            A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, p=p)
        )
        return self

    def to_gray(self, p=0.25) -> "Augmenter":
        self.__transforms.append(A.ToGray(p=p))
        return self

    def motion_blur(self, blur_limit=5, p=0.25) -> "Augmenter":
        self.__transforms.append(A.MotionBlur(blur_limit=blur_limit, p=p))
        return self

    def one_of_sharp_blur(self, p=0.25) -> "Augmenter":
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
            x_scale: tuple[float, float] = (0.85, 1.15),
            y_scale: tuple[float, float] = (0.95, 1.05),
            p=0.5,
            sub_p=0.25,
    ) -> "Augmenter":
        """ 拉伸组合增强 """
        self.__transforms.append(
            A.OneOf([
                A.Affine(scale={"x": x_scale}, interpolation=cv2.INTER_AREA, fit_output=True, p=sub_p),
                A.Affine(scale={"y": y_scale}, interpolation=cv2.INTER_AREA, fit_output=True, p=sub_p),
            ], p=p)
        )
        return self
