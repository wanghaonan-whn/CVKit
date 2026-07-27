import cv2
import numpy as np
from pathlib import Path
from cvkit.core.annotation.io.txt import TXTDocument
from cvkit.core.image.io import ImageIO


class MaskImageUtils(ImageIO):
    def __init__(
            self,
            image_path: str | Path,
            label_path: str | Path,
            cls: int | str = 0,
            erase_num: int = 0,
    ) -> None:
        super().__init__(image_path)
        self.erase_num = erase_num
        self.label_path = label_path
        self.cls = int(cls)
        if self.erase_num < 0:
            raise ValueError("erase_num must be >= 0")

    def generate(self) -> "MaskImageUtils":
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        lines = TXTDocument(self.label_path).readlines()
        if len(lines) == 0:
            raise ValueError("label_path must contain at least one line")
        for line in lines:
            values = line.strip().split()
            if len(values) < 7:
                raise ValueError(f"At least 7 lines in label_path: {self.label_path}")
            if (len(values) - 1) % 2 != 0:
                raise ValueError(f"label num must oushu")

            coordinates = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)

            coordinates[:, 0] *= self.width
            coordinates[:, 1] *= self.height

            coordinates[:, 0] = np.clip(coordinates[:, 0], 0, self.width - 1)
            coordinates[:, 1] = np.clip(coordinates[:, 1], 0, self.height - 1)

            polygon = np.rint(coordinates).astype(np.int32)
            cv2.fillPoly(mask, [polygon], 255)

        save_path = self.path.parents[1] / "mask" / self.path.name
        self.save(mask, save_path)
        return self
