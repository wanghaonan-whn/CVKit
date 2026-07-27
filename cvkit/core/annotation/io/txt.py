from __future__ import annotations
from pathlib import Path
from cvkit.core.annotation.io.abc.base import BaseDocument


class TXTDocument(BaseDocument):
    """
        TXT文件类
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.content = None

    def read(self) -> str:
        with open(str(self.path), "r", encoding="utf-8") as f:
            self.content = f.read()
        return self.content

    def __str__(self) -> str:
        return self.content or ""

    def __len__(self) -> int:
        return len(self.content or "")

    def readlines(self) -> list[str]:
        with open(self.path, "r", encoding="utf-8") as f:
            return f.readlines()

    def write(self, content: str) -> TXTDocument:
        self.content = content
        return self

    def append(self, content: str) -> TXTDocument:
        if self.content is None:
            self.read()
        self.content += content
        return self

    def save(self, save_path: str | Path | None = None) -> TXTDocument:
        if self.content is None:
            raise ValueError("content is None")

        path = Path(save_path) if save_path is not None else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.content)
        return self


if __name__ == "__main__":
    txt_path = r"D:\BaiduNetdiskDownload\Software-v7.5.1-c4180852-20251120\labels\Image00223_02 7c8e51c9-97c0-4886-b8ea-7be5762c0516.txt"
    txtdoc = TXTDocument(txt_path).write("").append("1").save()
