from pathlib import Path
from collections.abc import Mapping
from cvkit.core.annotation.annotation import YOLOAnnotationUtils
from cvkit.core.annotation.hbb.voc import VOCAnnotationUtils
from cvkit.core.annotation.utils import AnnotationUtils


class YOLODetectionUtils(YOLOAnnotationUtils):
    """
        YOLO 检测工具类 V1.1
    """

    def validate(self):
        for label in self.labels:
            if len(label) != 5:
                raise ValueError(f"Label {label} is invalid")
            class_id, x, y, width, height = label
            if width <= 0 or height <= 0:
                raise ValueError(f"Detection label has invalid size: {label}")
            if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
                raise ValueError(f"Detection label contains coordinates outside [0, 1]: {label}")

    def save_as_voc(
            self,
            img_name: str,
            img_size: tuple[int, int],
            classes_mapping: Mapping[int, str],
            save_path: str | Path,
            depth: int = 1
    ) -> "YOLODetectionUtils":
        save_path = Path(save_path)
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

        for class_id, x, y, box_width, box_height in self.labels:
            xmin, ymin, xmax, ymax = AnnotationUtils.yolo_to_voc(img_size, x, y, box_width, box_height)
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            document.append_object(class_name=classes_mapping[int(class_id)], bbox=bbox)

        document.save()
        return self
