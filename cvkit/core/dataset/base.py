from functools import cached_property
from pathlib import Path
from typing import Iterator


class Dataset:
    def __init__(self, data_dir: str | Path, image_ext: str = ".jpg") -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.label_dir = self.data_dir / "labels"
        self.image_ext = image_ext.lower()

    def find_image(self, stem: str) -> Path | None:
        """查找唯一匹配的图片；同名多个文件时抛出异常。"""
        matches = self._image_index.get(stem, [])
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"同名图片存在歧义：{stem!r}，匹配文件：{matches}")
        return matches[0]

    def iter_images(self) -> Iterator[Path]:
        for image_file in self.image_dir.iterdir():
            if not image_file.is_file():
                print(f"Skipping! {image_file} is not a file")
                continue
            yield image_file

    def iter_labels(self) -> Iterator[Path]:
        for label_file in self.label_dir.iterdir():
            if not label_file.is_file():
                print(f"Skipping! {label_file} is not a file")
                continue
            if label_file.suffix != ".txt":
                print(f"Skipping! {label_file} ext is {label_file.suffix}")
                continue
            yield label_file

    @cached_property
    def _image_index(self) -> dict[str, list[Path]]:
        """扫描一次目录，按文件名主干建立索引。"""
        index: dict[str, list[Path]] = {}
        for image_file in self.iter_images():
            index.setdefault(image_file.stem, []).append(image_file)
        return index

    def refresh_image_index(self) -> None:
        """图片新增、删除或重命名后，让下一次查找重新建立索引。"""
        self.__dict__.pop("_image_index", None)
