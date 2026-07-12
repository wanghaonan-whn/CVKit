from __future__ import annotations

from pathlib import Path

from pathkit.process.io.abc.base import BaseDocument


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

    def save(self) -> TXTDocument:
        if self.content is None:
            raise ValueError("content is None")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(self.content)
        return self


if __name__ == "__main__":
    txtdoc = TXTDocument("/mnt/8T/TF/木地板破损/赛马/From_ldm/datasets2/labels/00000_0000000_Kp7K4i_11_3_11.txt")
    print(txtdoc.read())
