from __future__ import annotations

import json
from pathlib import Path
from typing import List
from cvkit.core.annotation.io.xmldoc import XMLDocument


class VOCAnnotationUtils(XMLDocument):
    """
        XML 标签工具
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)

    def get_voc_names(self) -> list[str]:
        document = XMLDocument(self.path)
        return [
            node.text
            for node in document.findall("object/name")
            if node.text is not None
        ]

    def is_label_in_voc(self, keyword: str) -> bool:
        """关键词查找对应的xml文件"""
        if keyword in self.get_voc_names():
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
        document = XMLDocument(self.path)
        size = document.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        parse_list = []
        for node in document.findall("object"):
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

    def rename_voc_label(self, new_label: str, old_label: str) -> VOCAnnotationUtils:
        """重命名标签"""
        document = XMLDocument(self.path)
        for node in document.findall("object/name"):
            if node.text == old_label:
                node.text = new_label
        document.save()
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
            save_path = self.path.parent.joinpath("labels")
        else:
            save_path = Path(save_path)
        class_names = self.get_voc_names()
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
            save_path = self.path.parent.joinpath("json")
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
    voc_path = r"D:\datasets\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations\2007_000027.xml"
    vocutils = VOCAnnotationUtils(voc_path)
    name = vocutils.get_voc_names()
    print(name)
