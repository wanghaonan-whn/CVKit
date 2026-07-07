from __future__ import annotations

from pathlib import Path

from pathkit.process.io.abc.base import BaseDocument


class TXTDocument(BaseDocument):
    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.content = None

    def read(self) -> str:
        with open(str(self.path), "r", encoding="utf-8") as f:
            self.content = f.read()
        return self.content

    def readlines(self) -> list[str]:
        with open(str(self.path), "r", encoding="utf-8") as f:
            return f.readlines()

    def save(
            self,
            path: str | Path | None = None,
            force: bool = False,
    ) -> None:
        save_path = path if path else self.path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        if self.content is None:
            if not force:
                raise ValueError("content is None, call read() or assign content first.")
            self.content = ""

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(self.content)


if __name__ == "__main__":
    txtdoc = TXTDocument("/mnt/8T/TF/木地板破损/赛马/From_ldm/datasets2/labels/00000_0000000_Kp7K4i_11_3_11.txt")
    print(txtdoc.read())
