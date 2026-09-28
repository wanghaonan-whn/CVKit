from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class DatasetCheckResult:
    bad_images: list[Path] = field(default_factory=list)
    missing_labels: list[Path] = field(default_factory=list)
    empty_labels: list[Path] = field(default_factory=list)
    orphan_labels: list[Path] = field(default_factory=list)