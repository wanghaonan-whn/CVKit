import math
from pathlib import Path

from cvkit.core.annotation.hbb.yolo import YOLOAnnotationUtils


class CopyPaste:
    """
        贴图增强类
    """

    def __init__(
            self,
            image_path: str | Path,
            pattern_dir: str | Path,
            class_id: int = 0,
    ):
        self.class_id = class_id
        self.image_path = Path(image_path)
        self.label_path = self.image_path.parents[1] / "labels" / f"{self.image_path.stem}.txt"

    def paste(self):
        bboxes, classes = YOLOAnnotationUtils(self.label_path).parse_label()
        num_cls = classes.count(self.class_id)
        num_pase = math.ceil(num_cls / 2)


if __name__ == "__main__":
    CopyPaste(
        "/mnt/FourT/test/cachu/images/1.png",
        None,
    ).paste()
