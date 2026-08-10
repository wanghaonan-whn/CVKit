import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from typing import Literal

from typing_extensions import Self

from cvkit.core.image.abc.base import BaseImage


class ImageIO(BaseImage):
    def __init__(self, path):
        super().__init__(path)
        self.image = None
        self.save_path = None

    def read(
            self,
            load_mode: Literal["cv2", "PIL"] = "cv2",
            load_type: Literal["RGB", "L", "RGBA"] = "RGB",
    ) -> Self:
        cv_type = {
            "RGB": cv2.IMREAD_COLOR,
            "L": cv2.IMREAD_GRAYSCALE,
            "RGBA": cv2.IMREAD_UNCHANGED,
        }
        if load_mode == "cv2":
            self.image = cv2.imread(str(self.path), cv_type[load_type])
        elif load_mode == "PIL":
            self.image = Image.open(self.path).convert(load_type)
        else:
            raise ValueError(f"Unsupported load mode: {load_mode}")
        return self

    @property
    def width(self) -> int:
        if isinstance(self.image, Image.Image):
            return self.image.width
        if isinstance(self.image, np.ndarray):
            return self.image.shape[1]
        raise ValueError("Image has not been loaded")

    @property
    def height(self) -> int:
        if isinstance(self.image, Image.Image):
            return self.image.height
        if isinstance(self.image, np.ndarray):
            return self.image.shape[0]
        raise ValueError("Image has not been loaded")

    def save(self, image=None, save_path: str | Path | None = None):
        path = Path(save_path) if save_path is not None else self.path
        target = image if image is not None else self.image

        if target is None:
            raise ValueError("No image to save")
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(target, Image.Image):
            target.save(path)
        elif isinstance(target, np.ndarray):
            if not cv2.imwrite(str(path), target):
                raise IOError(f"Failed to save image: {path}")
        else:
            raise TypeError(f"Unsupported image type: {type(target)}")
        return self


if __name__ == "__main__":
    pass
