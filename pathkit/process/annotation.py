from typing import List

from pathkit.base.path import PathEntry, PathList
from pathkit.base.utils import PathUtils
from pathkit.process.xmldocument import XMLDocument


class AnnotationUtils:
    """
        XML 标签工具
    """

    @staticmethod
    def get_xml_label_names(xml_path: str | PathEntry) -> list[str]:
        """获取标注 XML 中所有 object/name 文本"""
        document = XMLDocument(xml_path)
        return [
            node.text
            for node in document.findall("object/name")
            if node.text is not None
        ]

    @staticmethod
    def get_xmls_label_names(xmls_path: str | PathEntry) -> list[str]:
        """获取标注 XML 文件夹中所有 object/name 文本"""
        xmls_path_list = PathUtils.glob_paths(xmls_path, "*.xml")
        label_names = [
            node.text
            for document in xmls_path_list
            for node in XMLDocument(document).findall("object/name")
            if node.text is not None
        ]
        return PathList(label_names).unique().to_str()

    @staticmethod
    def get_keyword_with_xml_label(src_path: str, keyword: str, is_recursion: bool = False) -> PathList:
        """关键词查找对应的xml文件"""
        file_paths = PathUtils.get_file_paths_with_suffix(src_path, suffix="xml", is_recursion=is_recursion)
        target_path = []
        for file_path in file_paths:
            if keyword in AnnotationUtils.get_xml_label_names(file_path):
                target_path.append(file_path)
        return PathList(target_path)

    @staticmethod
    def parse_xml_file(src_path: str | PathEntry) -> List:
        """解析xml标注文件"""
        document = XMLDocument(src_path)
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
        return parse_list

    @staticmethod
    def rename_xml_label():
        pass
