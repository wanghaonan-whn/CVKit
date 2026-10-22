import shutil
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

from cvkit.core.dataset.base import Dataset
from cvkit.core.dataset.result import DatasetCheckResult


class CleanType(str, Enum):
    BAD_IMAGES = "bad_images"
    ORPHAN_IMAGES = "orphan_images"
    EMPTY_LABELS = "empty_labels"
    ORPHAN_LABELS = "orphan_labels"
    INVALID_LABELS = "invalid_labels"
    REPEAT_IMAGES = "repeat_images"


CleanPlan = dict[CleanType, dict[Path, Path]]


class DatasetCleaner:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.quarantine_dir = dataset.data_dir / "quarantine"

    def build_plan(self, result: DatasetCheckResult) -> CleanPlan:
        """按问题类别构建源文件到隔离位置的映射，不修改文件。"""
        bad_images = list(dict.fromkeys(result.bad_images))
        bad_plan = self._build_image_plan(bad_images, CleanType.BAD_IMAGES, include_labels=True)

        reserved_files = set(bad_plan)
        repeat_images = [duplicate for _, duplicate in result.repeat_images if duplicate not in reserved_files]
        repeat_plan = self._build_image_plan(repeat_images, CleanType.REPEAT_IMAGES, include_labels=True)

        reserved_files.update(repeat_plan)
        orphan_image_plan = self._build_image_plan([path for path in result.orphan_images if path not in reserved_files], CleanType.ORPHAN_IMAGES)

        reserved_files.update(orphan_image_plan)
        orphan_label_plan = self._build_label_plan([path for path in result.orphan_labels if path not in reserved_files], CleanType.ORPHAN_LABELS)

        reserved_files.update(orphan_label_plan)
        invalid_label_plan = self._build_label_plan([path for path in result.invalid_labels if path not in reserved_files], CleanType.INVALID_LABELS)

        reserved_files.update(invalid_label_plan)
        empty_label_plan = self._build_label_plan([path for path in result.empty_labels if path not in reserved_files], CleanType.EMPTY_LABELS)

        return {
            CleanType.BAD_IMAGES: bad_plan,
            CleanType.ORPHAN_IMAGES: orphan_image_plan,
            CleanType.EMPTY_LABELS: empty_label_plan,
            CleanType.ORPHAN_LABELS: orphan_label_plan,
            CleanType.INVALID_LABELS: invalid_label_plan,
            CleanType.REPEAT_IMAGES: repeat_plan,
        }

    def clean(self, plan: CleanPlan, clean_type: CleanType | str, dry_run: bool = True) -> dict[Path, Path]:
        """预览或执行一种清理计划。"""
        selected_type = CleanType(clean_type)
        moves = dict(plan.get(selected_type, {}))
        if dry_run:
            return moves

        self._validate_moves(moves)
        for source, destination in moves.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, destination)
        return moves

    def _build_image_plan(self, image_files: list[Path], clean_type: CleanType, include_labels: bool = False) -> dict[Path, Path]:
        moves: dict[Path, Path] = {}
        for image_file in dict.fromkeys(image_files):
            moves[image_file] = self._destination(clean_type, "images", image_file.name)
            if include_labels:
                label_file = self.dataset.label_dir / f"{image_file.stem}.txt"
                if label_file.is_file():
                    moves[label_file] = self._destination(clean_type, "labels", label_file.name)
        return moves

    def _build_label_plan(self, label_files: Iterable[Path], clean_type: CleanType, ) -> dict[Path, Path]:
        return {
            label_file: self._destination(clean_type, "labels", label_file.name)
            for label_file in dict.fromkeys(label_files)
        }

    def _destination(self, clean_type: CleanType, data_type: str, name: str) -> Path:
        return self.quarantine_dir / clean_type.value / data_type / name

    @staticmethod
    def _validate_moves(moves: dict[Path, Path]) -> None:
        for source, destination in moves.items():
            if not source.is_file():
                raise FileNotFoundError(f"File not found: {source}")
            if destination.exists():
                raise FileExistsError(f"Quarantine file already exists: {destination}")
