from __future__ import annotations

from collections import Counter
from pathlib import Path

from pathkit.process.io.txt import TXTDocument


class YOLOAnnotationUtils(TXTDocument):
    """
        YOLO 标签工具
        修改类别
        合并标签
        删除类别
        统计类别
    """

    def change_cls(self, src_cls: str | int, dst_cls: str | int) -> YOLOAnnotationUtils:
        """ 修改类别 """
        src_cls = int(src_cls)
        dst_cls = int(dst_cls)

        new_lines = []

        for line in self.readlines():
            parts = line.split()

            if not parts:
                new_lines.append(line)
                continue

            if int(parts[0]) == src_cls:
                parts[0] = str(dst_cls)
            new_lines.append(" ".join(parts) + "\n")
        self.content = "".join(new_lines)
        self.save()
        return self

    def del_cls(self, src_cls: str | int) -> None:
        """ 删除类别 """
        src_cls = int(src_cls)

        new_lines = []
        for line in self.readlines():
            if int(line.split()[0]) == src_cls:
                continue
            new_lines.append(line)
        self.content = "".join(new_lines)
        self.save()

    def remove_empty(self) -> YOLOAnnotationUtils:
        readlines = self.readlines()
        if len(readlines) == 0:
            self.path.unlink()
        return self

    def remove_duplicate(self) -> YOLOAnnotationUtils:
        readlines = self.readlines()
        unique = list(dict.fromkeys(readlines))
        self.content = "\n".join(unique)
        self.save()
        return self

    @staticmethod
    def parse_label_classes(lines: list[str]) -> set:
        """ 单个yolo标注类别集合 """
        classes = []
        for line in lines:
            classes.append(int(line.split()[0]))

        return set(classes)

    @staticmethod
    def count_classes(label_dir: str | Path) -> Counter[int]:
        counter = Counter()

        label_dir = Path(label_dir)
        label_list = label_dir.glob("*.txt")
        for txt_file in label_list:
            with txt_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    cls = int(line.split()[0])
                    counter[cls] += 1

        return counter


if __name__ == "__main__":
    YOLOAnnotationUtils(
        r"D:\BaiduNetdiskDownload\Software-v7.5.1-c4180852-20251120\labels\Image00223_02 7c8e51c9-97c0-4886-b8ea-7be5762c0516.txt")
