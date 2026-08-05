from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing_extensions import Self
from cvkit.core.annotation.io.abc.base import BaseDocument


class XmlDocument(BaseDocument):
    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.__tree: ET.ElementTree | None = None
        self.__root: ET.Element | None = None

    def __str__(self):
        return ET.tostring(self.__root, encoding="unicode")

    @classmethod
    def new(cls, path: str | Path, root_tag: str = "root"):
        document = cls(path)
        document.__root = ET.Element(root_tag)
        document.__tree = ET.ElementTree(document.__root)
        return document

    def read(self) -> Self:
        self.__tree = ET.parse(self.path)
        self.__root = self.__tree.getroot()
        return self

    @property
    def root(self) -> ET.Element:
        return self.__root

    def find(self, xpath: str) -> ET.Element | None:
        """
            <size>
                <width>486</width>
                <height>500</height>
                <depth>3</depth>
            </size>
            Args:
                xpath: "size/width"
            Return: Element对象
        """
        return self.__root.find(xpath)

    def findall(self, xpath: str) -> list[ET.Element]:
        return self.__root.findall(xpath)

    def gettext(self, xpath: str) -> str | None:
        """获取节点文本"""
        node = self.find(xpath)
        return node.text if node is not None else None

    def getattr(self, xpath: str, attr_name: str, default: str | None = None) -> str | None:
        """
            <person id="1001" name="Tom">
            Args:
                xpath: "person"
                attr_name: "id"
            Returns:
                1001
        """
        node = self.find(xpath)
        return node.attrib.get(attr_name, default) if node is not None else default

    def update_text(self, xpath: str, new_value: str) -> XmlDocument:
        """更新节点文本"""
        node = self.find(xpath)
        if node is not None:
            node.text = new_value
        else:
            raise KeyError("Not found node {}".format(xpath))
        return self

    def update_attr(self, xpath: str, attr_name: str, value: str) -> XmlDocument:
        """更新节点属性"""
        node = self.find(xpath)
        if node is not None:
            node.attrib[attr_name] = value
        else:
            raise KeyError("Not found node {}".format(xpath))
        return self

    def append_node(self, xpath: str, tag: str, text: str | None = None, attrib: dict[str, str] | None = None) -> XmlDocument:
        """在指定节点下追加子节点"""
        parent = self.find(xpath)
        if parent is None:
            raise KeyError("Not found parent node {}".format(xpath))
        node = ET.Element(tag, attrib or {})
        node.text = text
        parent.append(node)
        return self

    def remove_node(self, xpath: str) -> XmlDocument:
        """删除匹配节点"""
        target = self.find(xpath)
        if target is None:
            raise KeyError("Not found need removed node {}".format(xpath))

        for parent in self.__root.iter():
            for child in list(parent):
                if child is target:
                    parent.remove(child)
                    return self
        if target is self.__root:
            raise ValueError("You should not remove root node")
        return self

    def save(self) -> Self:
        ET.indent(self.__tree, space="    ")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.__tree.write(self.path, encoding="utf-8", xml_declaration=True)
        return self


if __name__ == "__main__":
    # xml_path = r"D:\datasets\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations\2007_000027.xml"
    # xmldoc = XMLDocument(xml_path)
    # print(xmldoc.root.text)
    # print(xmldoc.read())
    # print(xmldoc.find("size/width").text)
    # print(xmldoc.gettext("size/height"))
    # print(xmldoc.getattr("size", "width"))
    # print(xmldoc.update_text("a", "1"))
    # print(xmldoc.append_node("a", "1"))
    # xmldoc.update_text("size/width", "486").save()
    xml_path = "/mnt/FourT/TV/内部测试/HSBK_成昆线成都上行_CR200J1B_20250223_141847_8/HSBK_成昆线成都上行_CR200J1B_20250223_141847_8/xml/1_1.xml"
    xml = XmlDocument(xml_path).read()
    print(xml)
