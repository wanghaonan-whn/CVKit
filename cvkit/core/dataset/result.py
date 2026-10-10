from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DuplicateBBoxIssue:
    label_file: Path
    first_index: int
    second_index: int
    iou: float


@dataclass
class DatasetCheckResult:
    bad_images: list[Path] = field(default_factory=list)
    orphan_images: list[Path] = field(default_factory=list)
    empty_labels: list[Path] = field(default_factory=list)
    orphan_labels: list[Path] = field(default_factory=list)
    repeat_labels: list[DuplicateBBoxIssue] = field(default_factory=list)  # 标注文件、框编号、框编号、IoU
    repeat_images: list[tuple[Path, Path]] = field(default_factory=list)
    invalid_labels: dict[Path, str] = field(default_factory=dict)
    ambiguous_images: dict[str, list[Path]] = field(default_factory=dict)
