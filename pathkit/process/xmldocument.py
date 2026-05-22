from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


class XMLDocument:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._tree = ET.parse(self.path)
        self._root = self._tree.getroot()

    def read(self) -> str:
        return ET.tostring(self._root, encoding="unicode")

    @property
    def root(self) -> ET.Element:
        return self._root

    def find(self, xpath: str) -> ET.Element | None:
        return self._root.find(xpath)

    def findall(self, xpath: str) -> list[ET.Element]:
        return self._root.findall(xpath)

    def get_text(self, xpath: str) -> str | None:
        """获取节点文本"""
        node = self.find(xpath)
        return node.text if node is not None else None

    def get_attr(self, xpath: str, attr_name: str, default: str | None = None) -> str | None:
        """获取节点属性"""
        node = self.find(xpath)
        return node.attrib.get(attr_name, default) if node is not None else default

    def update_text(self, xpath: str, new_value: str) -> bool:
        """更新节点文本"""
        node = self.find(xpath)
        if node is not None:
            node.text = new_value
            return True
        return False

    def update_attr(self, xpath: str, attr_name: str, value: str) -> bool:
        """更新节点属性"""
        node = self.find(xpath)
        if node is not None:
            node.attrib[attr_name] = value
            return True
        return False

    def append_node(self, xpath: str, tag: str, text: str | None = None, attrib: dict[str, str] | None = None) -> bool:
        """在指定节点下追加子节点，成功返回 True，未找到父节点返回 False"""
        parent = self.find(xpath)
        if parent is None:
            return False
        node = ET.Element(tag, attrib or {})
        node.text = text
        parent.append(node)
        return True

    def remove_node(self, xpath: str) -> bool:
        """删除匹配节点"""
        target = self.find(xpath)
        if target is None:
            return False

        for parent in self._root.iter():
            for child in list(parent):
                if child is target:
                    parent.remove(child)
                    return True
        if target is self._root:
            return False
        return False

    def save(self, path: str | Path | None = None) -> None:
        """保存 XML 到指定路径，默认覆盖原文件"""
        target = self.path if path is None else Path(path)
        ET.indent(self._tree, space="  ")
        self._tree.write(target, encoding="utf-8", xml_declaration=True)
