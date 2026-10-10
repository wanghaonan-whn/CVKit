from pathlib import Path
from typing import Iterator


class Dataset:
    IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.label_dir = self.data_dir / "labels"

    @staticmethod
    def find_image(stem: str, image_index: dict[str, list[Path]]) -> Path | None:
        """ 从索引查找唯一匹配的图片 """
        matches = image_index.get(stem, [])
        if len(matches) > 1:
            raise ValueError(f"同名图片存在歧义：{stem!r}，匹配文件：{matches}")
        return matches[0] if matches else None

    def iter_images(self) -> Iterator[Path]:
        for image_file in self.image_dir.iterdir():
            if not image_file.is_file():
                print(f"Skipping! {image_file} is not a file")
                continue
            if image_file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                # print(f"Skipping! {image_file.suffix} not in {self.IMAGE_EXTENSIONS}")
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

    def build_image_index(self) -> dict[str, list[Path]]:
        """ 为本次操作建立图片索引 """
        index: dict[str, list[Path]] = {}
        for image_file in self.iter_images():
            index.setdefault(image_file.stem, []).append(image_file)
        return index
