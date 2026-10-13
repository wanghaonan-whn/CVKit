"""使用标准库显示数据集处理阶段与完成数量。"""

import sys
import time
from collections.abc import Iterator, Sequence
from typing import TypeVar


T = TypeVar("T")


def iter_progress(
    items: Sequence[T],
    *,
    step: int,
    label: str,
    steps: int = 6,
    min_interval: float = 0.1,
) -> Iterator[T]:
    """每项完成后更新计数；终端原地刷新，IDE 或重定向输出每秒记录进度。"""
    stream = sys.stderr
    terminal = stream.isatty()
    total = len(items)
    interval = min_interval if terminal else max(min_interval, 1.0)

    def display(current: int) -> None:
        text = f"[{step}/{steps}] {label}：{current} / {total}"
        print(text, file=stream, end="\r" if terminal else "\n", flush=True)

    display(0)
    last_update = time.monotonic()
    try:
        for current, item in enumerate(items, start=1):
            yield item
            now = time.monotonic()
            if current == total or now - last_update >= interval:
                display(current)
                last_update = now
    finally:
        if terminal:
            print(file=stream, flush=True)
