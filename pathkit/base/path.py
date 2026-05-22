from __future__ import annotations
from collections import Counter
from pathlib import Path


class PathList(list):
    def parent(self):
        parents = [p.parent if isinstance(p, Path) else Path(p).parent for p in self]
        parents = list(dict.fromkeys(parents))  # 保序去重
        return PathList(parents)

    def to_str(self) -> list[str]:
        return [str(p) for p in self]

    def counter_suffixes(self) -> dict[str, int]:
        counter = Counter()
        for file in self:
            path = file if isinstance(file, Path) else Path(file)
            suffix = path.suffix.lstrip(".")
            if suffix:
                counter[suffix] += 1
        return dict(counter)

    def suffix_list(self) -> list[str]:
        return list(self.counter_suffixes().keys())

    def filter_file(self) -> "PathList":
        return PathList([item for item in self if (item if isinstance(item, Path) else Path(item)).is_file()])

    def filter_dir(self) -> "PathList":
        return PathList([item for item in self if (item if isinstance(item, Path) else Path(item)).is_dir()])

    def filter_exists(self) -> "PathList":
        return PathList([item for item in self if (item if isinstance(item, Path) else Path(item)).exists()])

    def sort_by_name(self, reverse: bool = False) -> "PathList":
        return PathList(
            sorted(self, key=lambda item: (item if isinstance(item, Path) else Path(item)).name, reverse=reverse))

    def sort_by_mtime(self, reverse: bool = False) -> "PathList":
        return PathList(
            sorted(self, key=lambda item: (item if isinstance(item, Path) else Path(item)).stat().st_mtime,
                   reverse=reverse))

    def unique(self) -> "PathList":
        normalized = []
        seen = set()
        for item in self:
            path = item if isinstance(item, Path) else Path(item)
            if path not in seen:
                seen.add(path)
                normalized.append(path)
        return PathList(normalized)
