import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from pathlib import Path
from cvkit.core.annotation.io.abc.base import BaseDocument


class TxtDocument(BaseDocument):
    """
        TXT文件类
    """

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        super().__init__(path)
        self.content: str | None = None
        self.encoding = encoding

    def __str__(self) -> str:
        return self.content or ""

    def __len__(self) -> int:
        return len(self.content or "")

    @classmethod
    def create(cls, path: str | Path, encoding: str = "utf-8") -> Self:
        doc = cls(path, encoding)
        doc.content = ""
        return doc

    def read(self) -> str | None:
        with open(self.path, "r", encoding=self.encoding) as f:
            self.content = f.read()
        return self.content

    def readlines(self, keepends: bool = True) -> list[str]:
        if self.content is None:
            self.read()
        return self.content.splitlines(keepends=keepends)

    def write(self, content: str) -> Self:
        """
            Replace the in-memory content.

            This method does not write to disk.
            Call save() to persist the content.
        """
        self.content = content
        return self

    def append(self, content: str) -> Self:
        if self.content is None:
            if self.path.exists():
                self.read()
            else:
                self.content = ""
        self.content += content
        return self

    def save(self, save_path: str | Path | None = None) -> Self:
        if self.content is None:
            raise ValueError("content is None")

        path = Path(save_path) if save_path is not None else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding=self.encoding) as f:
            f.write(self.content)
        return self


if __name__ == "__main__":
    txt_path = r"D:\BaiduNetdiskDownload\Software-v7.5.1-c4180852-20251120\labels\Image00223_02 7c8e51c9-97c0-4886-b8ea-7be5762c0516.txt"
    txtdoc = TxtDocument.create(txt_path).write("1").save()
