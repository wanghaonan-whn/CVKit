from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDocument(ABC):
    """
        文档抽象基类
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @abstractmethod
    def read(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def write(self, content: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path | None = None) -> None:
        raise NotImplementedError
