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
        self._remove_labels(result)
        return plan

    def _build_plan(self, result: DatasetCheckResult) -> list[str]:
        plan = []
        bad_dir = self.dataset.data_dir / "images_bad"
        for image_file in result.bad_images:
            save_path = bad_dir / image_file.name
            plan.append(f"MOVE: {image_file} -> {save_path}")
        label_files = dict.fromkeys(result.empty_labels + result.orphan_labels)
        for label_file in label_files:
            plan.append(f"DELETE LABEL: {label_file}")
        return plan

    def _move_bad_images(self, result: DatasetCheckResult) -> None:
        if not result.bad_images:
            return
        bad_dir = self.dataset.data_dir / "images_bad"
        bad_dir.mkdir(parents=True, exist_ok=False)
        for image_file in result.bad_images:
            shutil.move(image_file, bad_dir / image_file.name)

    @staticmethod
    def _remove_labels(result: DatasetCheckResult) -> None:
        label_files = dict.fromkeys(result.empty_labels + result.orphan_labels)
        for label_file in label_files:
            label_file.unlink()
