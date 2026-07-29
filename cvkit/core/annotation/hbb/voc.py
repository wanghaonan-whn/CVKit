from __future__ import annotations

import json
from pathlib import Path
from typing import List
from cvkit.core.annotation.io.xmlio import XMLDocument


class VOCAnnotationUtils(XMLDocument):
    """
        XML 标签工具
    """

    def get_voc_label_names(self) -> list[str]:
        return [
            node.text
            for node in self.findall("object/name")
            if node.text is not None
        ]

    def is_label_in_voc(self, keyword: str) -> bool:
        """关键词查找对应的xml文件"""
        if keyword in self.get_voc_label_names():
            return True
        else:
            return False

    def parse_voc(self) -> tuple[tuple, List]:
        """
            解析xml标注文件
            width: 宽
            height: 高
            parse_list: [xmin, ymin, xmax, ymax, 类名]
        """
        size = self.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        parse_list = []
        for node in self.findall("object"):
            name = node.find("name").text
            bbox = node.find("bndbox")
            if bbox is None:
                raise ValueError("bndbox is None")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            parse_list.append(
                [xmin, ymin, xmax, ymax, name]
            )
        return (width, height), parse_list

    @classmethod
    def build_annotation(
            cls, img_name: str, img_size: tuple[int, int], bbox: tuple[int, int, int, int],
            save_path: str | Path, class_name: str = "object", depth: int = 1,
    ) -> "VOCAnnotationUtils":
        width, height = img_size
        xmin, ymin, xmax, ymax = bbox
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
            .append_node(".", "object")
            .append_node("object", "name", text=class_name)
            .append_node("object", "pose", text="Unspecified")
            .append_node("object", "truncated", text="0")
            .append_node("object", "difficult", text="0")

            # add bndbox
            .append_node("object", "bndbox")
            .append_node("object/bndbox", "xmin", text=str(xmin))
            .append_node("object/bndbox", "ymin", text=str(ymin))
            .append_node("object/bndbox", "xmax", text=str(xmax))
            .append_node("object/bndbox", "ymax", text=str(ymax))
        )
        return document

    def rename_voc_label(self, new_label: str, old_label: str) -> VOCAnnotationUtils:
        """重命名标签"""
        for node in self.findall("object/name"):
            if node.text == old_label:
                node.text = new_label
        super().save()
        return self

    @staticmethod
    def voc_to_yolo(image_size, bbox) -> tuple:
        w, h = image_size
        xmin, ymin, xmax, ymax, name = bbox
        x = (xmin + xmax) / 2 / w
        y = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        return x, y, bw, bh

    def save_as_yolo(self, save_path: str | Path | None = None) -> VOCAnnotationUtils:
        if save_path is None:
            save_path = self.path.parents[1].joinpath("labels")
        else:
            save_path = Path(save_path)
        class_names = self.get_voc_label_names()
        class_id = {name: i for i, name in enumerate(sorted(set(class_names)))}

        yolo = []
        image_size, bboxes = self.parse_voc()
        for bbox in bboxes:
            x, y, bw, bh = VOCAnnotationUtils.voc_to_yolo(image_size, bbox)
            cls_id = class_id[bbox[4]]
            line = f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n"
            yolo.append(line)
        save_txt_path = save_path.joinpath(self.path.stem).with_suffix(".txt")
        save_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_txt_path, "w") as f:
            f.writelines(yolo)
        return self

    def save_as_json(
            self,
            save_path: str | Path | None = None,
            image_suffix: str = "jpg",
    ) -> VOCAnnotationUtils:
        if save_path is None:
            save_path = self.path.parents[1].joinpath("json")
        else:
            save_path = Path(save_path)

        json_content = {
            "version": "",
            "flags": {},
            "checked": False,
            "shapes": [],
            "imagePath": f"{self.path.stem}.{image_suffix}",
            "imageData": None,
        }
        shapes = []
        image_size, bboxes = self.parse_voc()
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox[:4]
            name = bbox[4]
            label = {
                "label": name,
                "shape_type": "rectangle",
                "flags": {},
                "points": [
                    [int(xmin), int(ymin)],
                    [int(xmax), int(ymin)],
                    [int(xmax), int(ymax)],
                    [int(xmin), int(ymax)],
                ],
                "group_id": None,
                "description": None,
                "difficult": False,
                "attributes": {},
            }
            shapes.append(label)

        json_content["shapes"] = shapes
        json_content["imageHeight"] = image_size[1]
        json_content["imageWidth"] = image_size[0]
        save_json_path = save_path / self.path.stem
        Path(save_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{save_json_path}.json", "w") as f:
            json.dump(json_content, f, ensure_ascii=False, indent=2)
        return self


if __name__ == "__main__":
    # voc_path = r"D:\datasets\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations\2007_000027.xml"
    # vocutils = VOCAnnotationUtils(voc_path)
    # name = vocutils.get_voc_label_names()
    # vocutils.save_as_yolo().save_as_json()
    # print(vocutils.parse_voc())
    # print(vocutils.get_voc_label_names())
    # vocutils.rename_voc_label("person1", "person")
    # print(vocutils.get_voc_label_names())
    # print(vocutils.is_label_in_voc("person1"))
    VOCAnnotationUtils.build_annotation("1.jpg", (100, 100), (2, 3, 4, 5), "./1.xml").save()
