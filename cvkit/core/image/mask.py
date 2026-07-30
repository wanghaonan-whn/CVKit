import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from cvkit.core.annotation.hbb.voc import VOCAnnotationUtils
from cvkit.core.annotation.io.txt import TXTDocument
from cvkit.core.image.io import ImageIO
from simple_lama_inpainting import SimpleLama


class MaskImageUtils(ImageIO):
    def __init__(
            self,
            image_path: str | Path,
            erase_num: int = 0,
            cls: int | str = 0,
    ) -> None:
        """
        :param image_path:
        :param label_path:
        :param cls: 擦除类别
        :param erase_num: 擦除个数
        """
        super().__init__(image_path)
        self.ori_image = self.read().image
        self.ori_image = cv2.cvtColor(self.ori_image, cv2.COLOR_BGR2RGB)
        self.erase_num = erase_num
        self.label_path = Path(image_path).parents[1] / "labels" / f"{Path(image_path).stem}.txt"
        self.cls = int(cls)
        if self.erase_num < 0:
            raise ValueError("erase_num must be >= 0")

    def save_mask_bbox(self, class_name: str = "object"):
        mask_bin = self.image
        save_dir = self.path.parents[1] / "cachu"
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("contours == 0")

        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) <= 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h
            bboxes.append([xmin, ymin, xmax, ymax])
        height, width = self.ori_image.shape[:2]

        save_xml_path = save_dir / "xml" / f"{self.path.stem}.xml"
        save_label_dir = save_dir / "labels"
        (
            VOCAnnotationUtils
            .build_annotation(
                f"{self.path.stem}.png",
                (width, height),
                bboxes,
                save_xml_path,
                class_name,
            )
            .save_as_yolo(save_label_dir)
        )
        return self

    def save_mask_seg(self):
        mask_bin = self.image
        save_dir = self.path.parents[1] / "cachu"
        height, width = self.ori_image.shape[:2]
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("contours == 0")

        labels = []
        for contour in contours:
            if cv2.contourArea(contour) <= 0:
                raise ValueError("contourArea <= 0")
            points = contour.reshape(-1, 2)
            coordinates = []
            for x, y in points:
                x_normalized = float(np.clip(x / width, 0.0, 1.0))
                y_normalized = float(np.clip(y / height, 0.0, 1.0))
                coordinates.append(f"{x_normalized:.6f}")
                coordinates.append(f"{y_normalized:.6f}")
            labels.append(f"{self.cls} {' '.join(coordinates)}")

        save_path = save_dir / "labels" / f"{self.path.stem}.txt"
        TXTDocument(save_path).write("\n".join(labels) + "\n").save()
        return self

    def expand_mask(self, dilate_kernel_size: int):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
        self.image = cv2.dilate(self.image, kernel, iterations=1)
        return self

    def repair(self):
        source = Image.fromarray(self.ori_image)
        mask = Image.fromarray(self.image).convert("L")
        self.image = SimpleLama()(source, mask)
        return self

    def generate_from_seg(self):
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

        self.image = mask
        save_path = self.path.parents[1] / "mask" / f"{self.path.stem}.png"
        super().save(mask, save_path)
        return self

    def generate_from_hbb(self):
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

        self.image = mask
        save_path = self.path.parents[1] / "mask" / f"{self.path.stem}.png"
        super().save(mask, save_path)
        return self


if __name__ == "__main__":
    from tqdm import tqdm

    image_dir = ""
    for image_path in tqdm(Path(image_dir).iterdir()):
        root = image_path.parents[1]
        result_path = root / "cachu" / "images" / f"{image_path.stem}.png"
        (
            MaskImageUtils(image_path, erase_num=1)
            .generate_from_seg()
            .save_mask_seg()
            .expand_mask(1)
            .repair()
            .save(save_path=result_path)
        )
