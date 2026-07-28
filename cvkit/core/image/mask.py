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
        """
        :param image_path:
        :param label_path:
        :param cls: 擦除类别
        :param erase_num: 擦除个数
        """
        super().__init__(image_path)
        self.erase_num = erase_num
        self.label_path = Path(label_path)
        self.cls = int(cls)
        if self.erase_num < 0:
            raise ValueError("erase_num must be >= 0")

    def generate(self) -> "MaskImageUtils":
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        lines = TXTDocument(self.label_path).readlines()
        if len(lines) == 0:
            raise ValueError("label_path must contain at least one line")

        polygons = []
        for line in lines:
            values = line.strip().split()
            if int(values[0]) != self.cls:
                continue
            if len(values) < 7:
                raise ValueError(f"At least 7 words in label_path: {self.label_path}")
            if (len(values) - 1) % 2 != 0:
                raise ValueError(f"label num must oushu")

            coordinates = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)

            coordinates[:, 0] *= self.width
            coordinates[:, 1] *= self.height

            coordinates[:, 0] = np.clip(coordinates[:, 0], 0, self.width - 1)
            coordinates[:, 1] = np.clip(coordinates[:, 1], 0, self.height - 1)

            polygons.append(np.rint(coordinates).astype(np.int32))

        if self.erase_num > len(polygons):
            raise ValueError(f"erase_num={self.erase_num} 超过标注数量 {len(polygons)}")
        rng = np.random.default_rng()
        selected_indices = rng.choice(len(polygons), size=self.erase_num, replace=False)
        for index in selected_indices:
            cv2.fillPoly(mask, [polygons[index]], 255)

        save_path = self.path.parents[1] / "mask" / f"{self.path.stem}.png"
        super().save(mask, save_path)
        return self
