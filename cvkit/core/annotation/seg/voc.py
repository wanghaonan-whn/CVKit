from pathlib import Path
from typing import List

from cvkit.core.annotation.io.xmlio import XmlDocument


class VOCSegmentUtils(XmlDocument):
    @classmethod
    def build_annotation(
            cls, img_name: str, img_size: tuple[int, int], bboxes: List[List[int]],
            save_path: str | Path, points, class_name: str = "object", depth: int = 1,
    ):
        width, height = img_size
        document = cls.create(save_path, root_tag="annotation")
        (
            document
            .append_node(".", "folder", text="images")
            .append_node(".", "filename", text=Path(img_name).name)
            .append_node(".", "path", text=img_name)
            .append_node(".", "source")
            .append_node("source", "database", text="Unknown")
            .append_node(".", "size")
            .append_node("size", "width", text=str(width))
            .append_node("size", "height", text=str(height))
            .append_node("size", "depth", text=str(depth))
            .append_node(".", "segmented", text="0")
        )

        for index, bbox in enumerate(bboxes, start=1):
            # TODO
            xmin, ymin, xmax, ymax = bbox
            document.append_node(".", "object")
            object_path = f"object[{index}]"
            bndbox_path = f"{object_path}/bndbox"
            (
                document
                .append_node(object_path, "name", text=class_name)
                .append_node(object_path, "pose", text="Unspecified")
                .append_node(object_path, "truncated", text="0")
                .append_node(object_path, "difficult", text="0")

                # add bndbox
                .append_node(object_path, "bndbox")
                .append_node(bndbox_path, "xmin", text=str(xmin))
                .append_node(bndbox_path, "ymin", text=str(ymin))
                .append_node(bndbox_path, "xmax", text=str(xmax))
                .append_node(bndbox_path, "ymax", text=str(ymax))
            )
        return document
