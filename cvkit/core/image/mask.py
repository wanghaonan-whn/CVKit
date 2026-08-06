import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing_extensions import Self
from cvkit.core.annotation.io.txt import TxtDocument
from cvkit.core.annotation.seg.yolo import YOLOSegmentationUtils
from cvkit.core.image.io import ImageIO
from simple_lama_inpainting import SimpleLama
from cvkit.core.annotation.hbb.yolo import YOLOAnnotationUtils


class MaskImageUtils(ImageIO):
    """
        Process: generate -> save/expand -> repair -> save_result
    """

    def __init__(
            self,
            image_path: str | Path,
            erase_num: int = 0,
            cls: int | str = 0,
    ) -> None:
        """
            :param image_path: path
            :param cls: 擦除类别
            :param erase_num: 擦除个数
        """
        super().__init__(image_path)
        if self.image is None:
            raise ValueError(f"Failed to read image: {self.path}")
        self.original_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.mask: np.ndarray | None = None
        self.result_image: Image.Image | np.ndarray | None = None
        self.erase_num = erase_num
        self.label_path = Path(image_path).parents[1] / "labels" / f"{Path(image_path).stem}.txt"
        self.cls = int(cls)
        if self.erase_num < 0:
            raise ValueError("erase_num must be >= 0")
        self.__lama = SimpleLama()

    def save_mask_bbox_as_yolo(self, save_path=None):
        height, width = self.original_image.shape[:2]
        lines = []

        for xmin, ymin, xmax, ymax in self.to_bboxes():
            x = (xmin + xmax) / 2 / width
            y = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            lines.append(f"{self.cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        save_path = self.path.parents[1] / "cachu/labels" / f"{self.path.stem}.txt" if save_path is None else save_path
        TxtDocument.new(save_path).write("\n".join(lines) + "\n").save()
        return self

    def save_mask_seg_as_yolo(self, save_path: str | Path | None = None):
        height, width = self.original_image.shape[:2]
        lines = []

        for polygon in self.to_polygons():
            coordinates = []
            for x, y in polygon:
                coordinates.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
            lines.append(f"{self.cls} {' '.join(coordinates)}")

        save_path = self.path.parents[1] / "cachu/labels" / f"{self.path.stem}.txt" if save_path is None else save_path
        TxtDocument.new(save_path).write("\n".join(lines) + "\n").save()
        return self

    def expand_mask(self, kernel_size: int):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.mask = cv2.dilate(self.mask, kernel, iterations=1)
        return self

    def repair(self):
        source = Image.fromarray(self.original_image)
        mask = Image.fromarray(self.mask).convert("L")
        self.result_image = self.__lama(source, mask)
        return self

    def generate_from_seg(self):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        lines = TxtDocument(self.label_path).readlines()
        annotations = YOLOSegmentationUtils.parse_label(lines)
        if len(lines) == 0:
            raise ValueError("label_path must contain at least one line")

        polygons = []
        for class_id, points in annotations:
            if class_id != self.cls:
                continue
            polygons.append(self.points_to_pixels(points))

        if self.erase_num > len(polygons):
            raise ValueError(f"erase_num={self.erase_num} 超过标注数量 {len(polygons)}")
        rng = np.random.default_rng()
        selected_indices = rng.choice(len(polygons), size=self.erase_num, replace=False)
        for index in selected_indices:
            cv2.fillPoly(mask, [polygons[index]], 255)

        self.mask = mask
        return self

    def generate_from_hbb(self):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        labels = YOLOAnnotationUtils(self.label_path).parse_label()
        if len(labels) == 0:
            raise ValueError("label cat not be empty")

        bboxes = []
        for class_id, x_center, y_center, box_width, box_height in labels:
            if class_id != self.cls:
                continue

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

        self.mask = mask
        return self

    def crop_mask_from_seg(self, save_dir: str | Path, image_index: int = 0) -> Self:
        height, width = self.shape[:2]
        lines = TxtDocument(self.label_path).readlines()
        annotations = YOLOSegmentationUtils.parse_label(lines)

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for index, (class_id, normalized_points) in enumerate(annotations):
            if class_id != self.cls: continue

            points = self.points_to_pixels(normalized_points)

            alpha = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(alpha, [points], 255)

            x, y, crop_width, crop_height = cv2.boundingRect(points)
            xmax = x + crop_width
            ymax = y + crop_height
            rgb_crop = self.original_image[y:ymax, x:xmax]
            alpha_crop = alpha[y:ymax, x:xmax]
            rgba_crop = np.dstack((rgb_crop, alpha_crop))
            super().save(Image.fromarray(rgba_crop), save_path=save_dir / f"crop_{self.path.stem}_{image_index}_{index}.png")
        return self

    def crop_mask_from_hbb(self, save_dir: str | Path, image_index: int = 0) -> Self:
        height, width = self.shape[:2]
        annotations = YOLOAnnotationUtils(self.label_path).parse_label()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for index, (class_id, *xywh) in enumerate(annotations):
            if class_id != self.cls: continue
            xmin, ymin, xmax, ymax = YOLOAnnotationUtils.yolo_to_voc((width, height), *xywh)
            xmin = int(round(xmin))
            ymin = int(round(ymin))
            xmax = int(round(xmax))
            ymax = int(round(ymax))

            rgb_crop = self.original_image[ymin:ymax, xmin:xmax]
            alpha_crop = np.full(rgb_crop.shape[:2], 255, dtype=np.uint8)
            rgba_crop = np.dstack((rgb_crop, alpha_crop))
            save_path = (save_dir / f"crop_{self.path.stem}_{image_index}_{index}.png")
            super().save(Image.fromarray(rgba_crop), save_path=save_path)
        return self

    def save_mask(self, save_path: str | Path | None = None) -> Self:
        save_path = self.path.parents[1] / "mask" / f"{self.path.stem}.png" if save_path is None else save_path
        super().save(self.mask, save_path)
        return self

    def save_result(self, save_path: str | Path) -> Self:
        super().save(self.result_image, save_path)
        return self

    def get_contours(self) -> list[np.ndarray]:
        mask = self.mask
        contours, _ = cv2.findContours(
            mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = [c for c in contours if cv2.contourArea(c) > 0]

        if not contours:
            raise ValueError("No valid contours found")
        return contours

    def to_bboxes(self) -> list[list[int]]:
        return [
            [x, y, x + width, y + height]
            for contour in self.get_contours()
            for x, y, width, height in [cv2.boundingRect(contour)]
        ]

    def to_polygons(self) -> list[np.ndarray]:
        return [
            contour.reshape(-1, 2)
            for contour in self.get_contours()
            if len(contour.reshape(-1, 2)) >= 3
        ]

    def points_to_pixels(self, points: list[tuple[float, float]]) -> np.ndarray:
        coordinates = np.asarray(points, dtype=np.float32)

        coordinates[:, 0] *= self.width
        coordinates[:, 1] *= self.height
        coordinates[:, 0] = np.clip(coordinates[:, 0], 0, self.width - 1)
        coordinates[:, 1] = np.clip(coordinates[:, 1], 0, self.height - 1)

        return np.rint(coordinates).astype(np.int32)


if __name__ == "__main__":
    # pass
    from tqdm import tqdm

    image_dir = ""
    for image_path in tqdm(Path(image_dir).iterdir()):
        root = image_path.parents[1]
        result_path = root / "cachu" / "images" / f"{image_path.stem}.png"
        (
            MaskImageUtils(image_path, erase_num=1)
            .read("cv2", "L")
            .generate_from_seg()
            .save_mask_seg_as_yolo()
            .expand_mask(1)
            .repair()
            .save(save_path=result_path)
        )
