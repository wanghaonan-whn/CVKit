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
        return self

    @property
    def shape(self):
        if isinstance(self.image, Image.Image):
            return self.image.size
        elif isinstance(self.image, np.ndarray):
            return self.image.shape
        else:
            raise TypeError("Image type not supported.")

    def save(self, image=None, save_path: str | Path | None = None):
        path = Path(save_path) if save_path is not None else self.path
        image = image if image is not None else self.image
        if isinstance(image, Image.Image):
            image = np.array(image)
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # 如果是 RGBA
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        return self


if __name__ == "__main__":
    pass
