import cv2
from pathlib import Path

import numpy as np

from cvkit.core.image.abc.base import BaseImage


class ImageIO(BaseImage):
    def __init__(self, path):
        super().__init__(path)
        self.image = None
        self.save_path = None

    def read(self) -> "ImageIO":
        self.image = cv2.imread(str(self.path))
        return self

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    def save(self, image=None, save_path: str | Path | None = None):
        path = Path(save_path) if save_path is not None else self.path
        image = image if image is not None else self.image
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        return self


if __name__ == "__main__":
    pass
