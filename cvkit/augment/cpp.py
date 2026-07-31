import math
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from cvkit.augment.aug import Augmenter
from cvkit.core.annotation.hbb.yolo import YOLOAnnotationUtils
from cvkit.core.annotation.io.txt import TXTDocument


class CopyPaste:
    """
        贴图增强类
    """

    def __init__(
            self,
            image_path: str | Path,
            class_id: int = 0,
    ):
        self.class_id = class_id
        self.image_path = Path(image_path)
        self.label_path = self.image_path.parents[1] / "labels" / f"{self.image_path.stem}.txt"
        self.save_path = self.image_path.parents[1] / "cp4qq1"

    def paste_with_bbox(self, n, pases: List, scale: tuple[float, float] = (1.0, 1.1)) -> None:
        lines = YOLOAnnotationUtils(self.label_path).get_classes_box(self.class_id)
        num_pase = math.ceil(len(lines) / 2)
        labels = YOLOAnnotationUtils.parse_label(lines)
        image_bg = Image.open(self.image_path)
        width, height = image_bg.size

        save_image_path = self.save_path / "images"
        save_label_path = self.save_path / "labels"
        save_image_path.mkdir(parents=True, exist_ok=True)
        save_label_path.mkdir(parents=True, exist_ok=True)

        txt = TXTDocument.create(save_label_path / f'guoche-{n}-{self.image_path.stem}-{self.class_id}-nump{num_pase}.txt')
        for _ in range(num_pase):
            label_choice = random.choice(labels)
            labels.remove(label_choice)
            x, y, w, h = YOLOAnnotationUtils.yolo_to_xy((width, height), *label_choice[1:])
            if x < 10 or y < 10 or x > width - 10 or y > height - 10:
                continue

            pattern = random.choice(pases)
            img_xpn = Image.open(pattern)
            r1 = random.uniform(*scale)

            w = max(10, w)
            h = max(10, h)
            w = min(13, w)
            h = min(13, h)

            p1 = round(w * r1)
            p2 = round(h * r1)

            img_xpn = img_xpn.resize((p1, p2))
            img_xpn = self.augment_pase(pattern, img_xpn)
            w_p, h_p = img_xpn.size
            x_p = x - (w_p / 2)
            y_p = y - (h_p / 2)

            image_bg.paste(img_xpn, (round(x_p), round(y_p)), img_xpn)

            txt.append(f"0 {label_choice[1]} {label_choice[2]} {label_choice[3]} {label_choice[4]}\n")

        txt.save()
        image_np = np.array(image_bg)
        transform_ori = (
            Augmenter(self.image_path)
            .brightness()
            .gauss_noise()
            .build()
        )
        augmented_image_np = transform_ori(image=image_np)['image']
        img_xpn = Image.fromarray(augmented_image_np)
        img_xpn.save(save_image_path / f'guoche-{n}-{self.image_path.stem}-{self.class_id}-nump{num_pase}.png')

    @staticmethod
    def convert_coords_to_yolo(class_id, xmin, ymin, xmax, ymax, img_width, img_height):
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        width = xmax - xmin
        height = ymax - ymin

        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height

        box = [class_id, x_center_norm, y_center_norm, width_norm, height_norm]
        return box

    @staticmethod
    def augment_pase(pattern, img_xpn):
        # transform = (
        #     Augmenter(pattern)
        #     .one_of_sharp_blur()
        #     .brightness()
        #     .gauss_noise()
        #     .build()
        # )

        img_pase = np.array(img_xpn)
        alpha = img_pase[..., -1]
        contours, _ = cv2.findContours(alpha, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        img_pase = img_pase[y:y + h, x:x + w, :]

        alpha = img_pase[..., -1]
        # augmented_image_np = transform(image=img_pase[..., :-1])['image']
        augmented_image_np = np.dstack((img_pase[..., :-1], alpha))

        img_xpn = Image.fromarray(augmented_image_np)
        return img_xpn


if __name__ == "__main__":
    pattern_path = Path("/mnt/FourT/pattern专用TV")
    pases = [p for p in pattern_path.rglob("*.png")]
    CopyPaste("/mnt/FourT/test/cachu/images/1.png").paste_with_bbox(10, pases)
