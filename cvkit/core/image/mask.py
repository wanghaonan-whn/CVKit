import cv2
import numpy as np
from pathlib import Path
from cvkit.core.annotation.io.txt import TXTDocument
from cvkit.core.image.io import ImageIO


class MaskImageUtils(ImageIO):
    def __init__(
            self,
            image_path: str | Path,
            label_path: str | Path | None = None,
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
        self.label_path = Path(label_path) if label_path else None
        self.cls = int(cls)
        if self.erase_num < 0:
            raise ValueError("erase_num must be >= 0")

    def find_largest_bbox(self) -> tuple[int, int, int, int] | None:
        mask_bin = ImageIO(self.path).read().image
        mask_bin = cv2.cvtColor(mask_bin, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        max_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnt)
        xmin, ymin = x, y
        xmax, ymax = x + w, y + h
        return xmin, ymin, xmax, ymax

    def generate_from_seg(self) -> "MaskImageUtils":
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        lines = TXTDocument(self.label_path).readlines()
        if len(lines) == 0:
            raise ValueError("label_path must contain at least one line")

        polygons = []
        for line in lines:
            values = line.strip().split()
            if int(values[0]) != self.cls: continue
            if len(values) < 7:
                raise ValueError(f"At least 7 words in label_path: {self.label_path}")
            if (len(values) - 1) % 2 != 0:
                raise ValueError(f"label num must OuShu")

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

    def generate_from_hbb(self) -> "MaskImageUtils":
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        lines = TXTDocument(self.label_path).readlines()
        if len(lines) == 0:
            raise ValueError("label cat not be empty")

        bboxes = []
        for line in lines:
            values = line.strip().split()
            class_id, x_center, y_center, box_width, box_height = values
            if int(class_id) != self.cls: continue

            x_center, y_center, box_width, box_height = float(x_center), float(y_center), float(box_width), float(box_height)

            x1 = round((x_center - box_width / 2) * self.width)
            y1 = round((y_center - box_height / 2) * self.height)
            x2 = round((x_center + box_width / 2) * self.width)
            y2 = round((y_center + box_height / 2) * self.height)

            x1 = int(np.clip(x1, 0, self.width - 1))
            y1 = int(np.clip(y1, 0, self.height - 1))
            x2 = int(np.clip(x2, 0, self.width - 1))
            y2 = int(np.clip(y2, 0, self.height - 1))

            bboxes.append(((x1, y1), (x2, y2)))

        if self.erase_num > len(bboxes):
            raise ValueError(f"erase_num={self.erase_num} 超过标注数量 {len(bboxes)}")
        rng = np.random.default_rng()
        selected_indices = rng.choice(len(bboxes), size=self.erase_num, replace=False)
        for index in selected_indices:
            top_left, bottom_right = bboxes[index]
            cv2.rectangle(mask, top_left, bottom_right, color=255, thickness=-1)

        save_path = self.path.parents[1] / "mask" / f"{self.path.stem}.png"
        super().save(mask, save_path)
        return self


if __name__ == "__main__":
    pass
