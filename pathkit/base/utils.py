from __future__ import annotations
from pathlib import Path
from typing import Iterable
from pathkit.base.path import PathList


class PathUtils:
    """
        路径工具
    """

    @staticmethod
    def __ensure_src_path_exists(src_path: str | Path) -> Path:
        path = Path(src_path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path

    @staticmethod
    def __collect_paths(iterator: Iterable[Path], on_permission_error: str = "skip") -> PathList:
        try:
            return PathList(list(iterator))
        except PermissionError:
            if on_permission_error == "skip":
                return PathList([])
            raise

    # 路径匹配
    @staticmethod
    def match_paths(
            src_path: str | Path,
            pattern: str = "*",
            is_recursion: bool = False,
            on_permission_error: str = "skip"
    ) -> PathList:
        """获取路径下匹配模式的路径列表"""
        path = PathUtils.__ensure_src_path_exists(src_path)
        if is_recursion:
            return PathUtils.__collect_paths(path.rglob(pattern), on_permission_error=on_permission_error)
        return PathUtils.__collect_paths(path.glob(pattern), on_permission_error=on_permission_error)

    @staticmethod
    def iter_files(src_path: str, is_recursion: bool = False, on_permission_error: str = "skip") -> PathList:
        """
            获取路径下所有文件列表
            src_path: 目标路径
            is_recursion: 是否递归获取子目录下的文件
            on_permission_error: 权限错误处理方式，"skip"表示跳过
        """
        file_paths = PathUtils.match_paths(
            src_path,
            is_recursion=is_recursion,
            on_permission_error=on_permission_error,
        )
        return file_paths.filter_file()

    @staticmethod
    def iter_dirs(src_path: str, is_recursion: bool = False, on_permission_error: str = "skip") -> PathList:
        """获取路径下所有目录列表"""
        dir_paths = PathUtils.match_paths(
            src_path,
            is_recursion=is_recursion,
            on_permission_error=on_permission_error,
        )
        return dir_paths.filter_dir()

    @staticmethod
    def filter_name(
            src_path: str,
            keyword: str,
            is_recursion: bool = False,
            include_dirs: bool = False,
            on_permission_error: str = "skip",
    ) -> PathList:
        """
            通过关键词过滤获取所有文件路径下所有符合的文件
            src_path: 父路径
            include_dirs: 是否包含文件夹
        """
        if include_dirs:
            paths = PathUtils.match_paths(
                src_path,
                is_recursion=is_recursion,
                on_permission_error=on_permission_error,
            )
            return PathList([item for item in paths if keyword in item.name])
        return PathList([
            item
            for item in PathUtils.iter_files(
                src_path,
                is_recursion=is_recursion,
                on_permission_error=on_permission_error,
            )
            if keyword in item.name
        ])

    @staticmethod
    def get_file_paths_with_suffixes(
            src_path: str,
            suffixes: list[str],
            is_recursion: bool = False,
            on_permission_error: str = "skip",
    ) -> PathList:
        normalized_suffixes = {suffix.lstrip(".").lower() for suffix in suffixes}
        file_paths = PathUtils.iter_files(
            src_path,
            is_recursion=is_recursion,
            on_permission_error=on_permission_error,
        )
        return PathList([
            file_path
            for file_path in file_paths
            if file_path.suffix.lstrip(".").lower() in normalized_suffixes
        ])

    @staticmethod
    def parse_file_with_suffix(
            src_path: str,
            is_recursion: bool = False,
            include_empty: bool = False,
            on_permission_error: str = "skip",
    ) -> list[str]:
        """获取路径下所有文件后缀"""
        file_paths = PathUtils.iter_files(
            src_path,
            is_recursion=is_recursion,
            on_permission_error=on_permission_error,
        )
        suffix = {file.suffix.lstrip(".") for file in file_paths}
        if not include_empty:
            suffix.discard("")
        return sorted(suffix)
