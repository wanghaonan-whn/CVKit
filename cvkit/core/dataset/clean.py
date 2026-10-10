import shutil
from cvkit.core.dataset.base import Dataset
from cvkit.core.dataset.result import DatasetCheckResult


class DatasetCleaner:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def clean(self, result: DatasetCheckResult, dry_run: bool = True) -> list[str]:
        plan = self._build_plan(result)
        if dry_run:
            return plan
        self._move_bad_images(result)
        self._remove_empty_labels(result)
        self._remove_orphan_labels(result)
        return plan

    def _build_plan(self, result: DatasetCheckResult) -> list[str]:
        plan = []
        bad_dir = self.dataset.data_dir / "images_bad"
        for image_file in result.bad_images:
            save_path = bad_dir / image_file.name
            plan.append(f"MOVE: {image_file} -> {save_path}")
        for label_file in result.empty_labels:
            plan.append(f"DELETE EMPTY LABEL: {label_file}")
        for label_file in result.orphan_labels:
            plan.append(f"DELETE ORPHAN LABEL: {label_file}")
        return plan

    def _move_bad_images(self, result: DatasetCheckResult) -> None:
        bad_dir = self.dataset.data_dir / "images_bad"
        bad_dir.mkdir(parents=True, exist_ok=True)

        for image_file in result.bad_images:
            save_path = bad_dir / image_file.name
            shutil.move(image_file, save_path)

    @staticmethod
    def _remove_empty_labels(result: DatasetCheckResult) -> None:
        for label_file in result.empty_labels:
            label_file.unlink()

    @staticmethod
    def _remove_orphan_labels(result: DatasetCheckResult) -> None:
        for label_file in result.orphan_labels:
            label_file.unlink()
