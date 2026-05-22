from __future__ import annotations
from pathlib import Path
from typing import List
from pathkit.base.path import PathList
from pathkit.base.utils import PathUtils
from pathkit.process.xmldocument import XMLDocument


class AnnotationUtils:
    """
        XML 标签工具
    """

    @staticmethod
    def get_xml_label_names(file_path: str | Path) -> list[str]:
        """获取标注 XML 中所有 object/name 文本"""
        document = XMLDocument(file_path)
        return [
            node.text
            for node in document.findall("object/name")
            if node.text is not None
        ]

    @staticmethod
    def get_xmls_label_names(xml_path: str | Path) -> list[str]:
        """获取标注 XML 文件夹中所有 object/name 文本"""
        xmls_path_list = PathUtils.match_path(xml_path, "*.xml")
        label_names = [
            node.text
            for document in xmls_path_list
            for node in XMLDocument(document).findall("object/name")
            if node.text is not None
        ]
        return PathList(label_names).to_str()

    @staticmethod
    def get_keyword_with_xml_label(src_path: str, keyword: str, is_recursion: bool = False) -> PathList:
        """关键词查找对应的xml文件"""
        file_paths = PathUtils.get_file_paths_with_suffixes(src_path, suffixes=["xml"], is_recursion=is_recursion)
        target_path = []
        for file_path in file_paths:
            if keyword in AnnotationUtils.get_xml_label_names(file_path):
                target_path.append(file_path)
        return PathList(target_path)

    @staticmethod
    def parse_xml_file(file_path: str | Path) -> tuple[tuple, List]:
        """
            解析xml标注文件
            width: 宽
            height: 高
            parse_list: [xmin, ymin, xmax, ymax, 类名]
        """
        document = XMLDocument(file_path)
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

    @staticmethod
    def rename_xml_label(file_path: str | Path, new_label: str, old_label: str) -> None:
        """重命名标签"""
        document = XMLDocument(file_path)
        for node in document.findall("object/name"):
            if node.text == old_label:
                node.text = new_label
        document.save(file_path)

    @staticmethod
    def voc_to_yolo(image_size, bbox) -> tuple:
        w, h = image_size
        xmin, ymin, xmax, ymax, name = bbox
        x = (xmin + xmax) / 2 / w
        y = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        return x, y, bw, bh

    def save_yolo_txt(self, xml_path: str | Path, save_path: str | Path | None = None) -> None:
        if save_path is None:
            save_path = Path(xml_path).parent.joinpath("labels")
        else:
            save_path = Path(save_path)
        xml_path_list = PathUtils.match_path(xml_path, "*.xml")
        class_names = self.get_xmls_label_names(xml_path)
        class_id = {name: i for i, name in enumerate(sorted(set(class_names)))}

        for file in xml_path_list:
            yolo = []
            image_size, bboxes = self.parse_xml_file(file)
            for bbox in bboxes:
                x, y, bw, bh = self.voc_to_yolo(image_size, bbox)
                cls_id = class_id[bbox[4]]
                line = f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n"
                yolo.append(line)
            save_txt_path = save_path.joinpath(file.stem).with_suffix(".txt")
            save_txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_txt_path, "w") as f:
                f.writelines(yolo)
