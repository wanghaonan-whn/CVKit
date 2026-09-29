from pathlib import Path
from typing import List
from collections.abc import Mapping
from cvkit.core.annotation.annotation import YOLOAnnotationUtils
from cvkit.core.annotation.hbb.voc import VOCAnnotationUtils
from cvkit.core.annotation.utils import AnnotationUtils


class YOLODetectionUtils(YOLOAnnotationUtils):
    """
        YOLO 检测工具类 V1.1
    """

    @classmethod
    def build(cls, labels: List[List[int | float]], save_path: str | Path) -> "YOLODetectionUtils":
        document = cls.new(save_path)
        for class_id, x, y, w, h in labels:
            document.append(f"{int(class_id)} {float(x):.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f}\n")
        return document

    def save_as_voc(
            self,
            img_name: str,
            img_size: tuple[int, int],
            classes_mapping: Mapping[int, str],
            save_dir: str | Path | None = None,
            depth: int = 1
    ) -> "YOLODetectionUtils":
        if save_dir is None:
            save_dir = self.path.parents[1].joinpath("xml")
        else:
            save_dir = Path(save_dir)

        save_path = save_dir / f"{self.path.stem}.xml"
        document = VOCAnnotationUtils.build(
            img_name=img_name,
            img_size=img_size,
            bboxes=[],
            save_path=save_path,
            depth=depth,
        )
        unknown_class_ids = set(self.count().keys()) - set(classes_mapping)
        if unknown_class_ids:
            raise ValueError(f"Classes missing from classes_mapping: {sorted(unknown_class_ids)}")

        labels = self.parse()
        for class_id, x, y, box_width, box_height in labels:
            xmin, ymin, xmax, ymax = AnnotationUtils.yolo_to_voc(img_size, x, y, box_width, box_height)
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            document.append_object(class_name=classes_mapping[int(class_id)], bbox=bbox)

        document.save()
        return self
