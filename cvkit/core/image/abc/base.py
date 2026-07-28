from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseImage(ABC):
    """
        图像抽象基类
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @abstractmethod
    def read(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> Any:
        raise NotImplementedError
