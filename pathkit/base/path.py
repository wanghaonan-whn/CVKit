import os
from collections import Counter
from pathlib import Path
from typing import List, Union


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
            sorted(self, key=lambda item: (item if isinstance(item, PathEntry) else PathEntry(item)).stat().st_mtime,
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


class PathEntry:
    """
        路径语义处理，对 pathlib.Path 类的基本包装
    """

    def __init__(self, path: Union[str, Path, "PathEntry"]):
        self.path = path.path if isinstance(path, PathEntry) else Path(path)

    def __str__(self):
        return str(self.path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path!s})"

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def joinpath(self, *args: Union[str, Path]) -> "PathEntry":
        """路径拼接"""
        if not args:
            raise ValueError("joinpath() requires at least one path segment")
        return PathEntry(self.path.joinpath(*args))

    def child(self, *others: Union[str, Path, "PathEntry"]) -> "PathEntry":
        return self.joinpath(*others)

    def normalize(self) -> "PathEntry":
        """规范化分隔符"""
        return PathEntry(os.path.normpath(str(self.path)))

    def absolute(self) -> "PathEntry":
        """基路径 -> 绝对路径"""
        return PathEntry(self.path.resolve())

    def relative_to(self, other: Union[str, Path, "PathEntry"]) -> "PathEntry":
        """相对路径"""
        base = other.path if isinstance(other, PathEntry) else Path(other)
        try:
            return PathEntry(self.path.relative_to(base))
        except ValueError as exc:
            raise ValueError(f"{self.path} is not under base path {base}") from exc

    def relative_other(self, other: Union[str, Path, "PathEntry"]) -> "PathEntry":
        return self.relative_to(other)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def parent(self) -> "PathEntry":
        """父路径"""
        return PathEntry(self.path.parent)

    @property
    def stem(self) -> str:
        """去扩展名文件名"""
        return self.path.stem

    @property
    def suffix(self) -> str:
        """扩展名"""
        return self.path.suffix

    @property
    def suffixes(self) -> List[str]:
        """all 扩展名"""
        return self.path.suffixes

    @property
    def parts(self) -> tuple[str, ...]:
        return self.path.parts

    @property
    def parents(self) -> PathList:
        """所有父路径"""
        return PathList([PathEntry(parent) for parent in self.path.parents])

    def with_suffix(self, suffix: str) -> "PathEntry":
        """修改扩展名"""
        return PathEntry(self.path.with_suffix(suffix))

    def with_name(self, name: str) -> "PathEntry":
        """修改文件名"""
        return PathEntry(self.path.with_name(name))

    def is_absolute(self) -> bool:
        """绝对路径"""
        return self.path.is_absolute()

    def common_path(self, *others: Union[str, Path, "PathEntry"]) -> "PathEntry":
        """共同父路径"""
        paths = [str(self.path)]
        for other in others:
            paths.append(
                str(other.path) if isinstance(other, PathEntry) else str(Path(other))
            )
        return PathEntry(os.path.commonpath(paths))

    def matches(self, pattern: str) -> bool:
        """glob匹配"""
        return self.path.match(pattern)

    def exists(self) -> bool:
        return self.path.exists()

    def is_file(self) -> bool:
        return self.path.is_file()

    def is_dir(self) -> bool:
        return self.path.is_dir()

    def is_symlink(self) -> bool:
        return self.path.is_symlink()

    def stat(self) -> os.stat_result:
        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")
        return self.path.stat()

    def as_posix(self) -> str:
        return self.path.as_posix()

    def as_uri(self) -> str:
        return self.path.as_uri()

    def expanduser(self) -> "PathEntry":
        return PathEntry(self.path.expanduser())

    def samefile(self, other: Union[str, Path, "PathEntry"]) -> bool:
        other_path = other.path if isinstance(other, PathEntry) else Path(other)
        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")
        if not other_path.exists():
            raise FileNotFoundError(f"Path does not exist: {other_path}")
        return self.path.samefile(other_path)

    def mkdir(self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        self.path.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

    def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
        """创建文件"""
        self.path.touch(mode=mode, exist_ok=exist_ok)

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.path.read_text(encoding=encoding)

    def write_text(self, data: str, encoding: str = "utf-8") -> int:
        return self.path.write_text(data, encoding=encoding)

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def write_bytes(self, data: bytes) -> int:
        return self.path.write_bytes(data)

    def unlink(self) -> None:
        """删除文件"""
        self.path.unlink()

    def rename(self, target: Union[str, Path, "PathEntry"]) -> "PathEntry":
        """重命名文件或移动文件，等价于mv"""
        target_path = target.path if isinstance(target, PathEntry) else Path(target)
        return PathEntry(self.path.rename(target_path))

    @property
    def drive(self) -> str:
        return self.path.drive

    @property
    def anchor(self) -> str:
        return self.path.anchor
