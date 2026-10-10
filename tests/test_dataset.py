import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cvkit.dataset import (
    Dataset,
    DatasetChecker,
    DatasetCleaner,
    DatasetCheckResult,
    DatasetSplitter,
    DatasetStatistics,
)
from cvkit.core.dataset.group import DatasetGroupChecker


class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def make_dataset(self, name: str = "dataset") -> Dataset:
        data_dir = self.root / name
        (data_dir / "images").mkdir(parents=True)
        (data_dir / "labels").mkdir(parents=True)
        return Dataset(data_dir)

    def write_image(
            self,
            path: Path,
            value: int,
            size: tuple[int, int] = (16, 12),
    ) -> Path:
        width, height = size
        image = np.full((height, width, 3), value, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image), f"Failed to create test image: {path}")
        return path

    @staticmethod
    def write_label(path: Path, content: str = "0 0.5 0.5 0.2 0.2\n") -> Path:
        path.write_text(content, encoding="utf-8")
        return path

    def test_dataset_iterators_index_and_find_image(self) -> None:
        dataset = self.make_dataset()
        jpg = self.write_image(dataset.image_dir / "same.jpg", 10)
        png = self.write_image(dataset.image_dir / "same.png", 20)
        upper = self.write_image(dataset.image_dir / "upper.jpeg", 30)
        upper.rename(dataset.image_dir / "upper.JPEG")
        (dataset.image_dir / "ignored.json").write_text("{}", encoding="utf-8")
        (dataset.image_dir / "subdir").mkdir()

        label = self.write_label(dataset.label_dir / "same.txt")
        self.write_label(dataset.label_dir / "ignored.TXT")
        (dataset.label_dir / "subdir").mkdir()

        with patch("builtins.print"):
            images = set(dataset.iter_images())
            labels = set(dataset.iter_labels())
            image_index = dataset.build_image_index()

        self.assertEqual(images, {jpg, png, dataset.image_dir / "upper.JPEG"})
        self.assertEqual(labels, {label})
        self.assertEqual(set(image_index["same"]), {jpg, png})
        self.assertEqual(
            dataset.find_image("upper", image_index),
            dataset.image_dir / "upper.JPEG",
        )
        self.assertIsNone(dataset.find_image("missing", image_index))
        with self.assertRaisesRegex(ValueError, "同名图片存在歧义"):
            dataset.find_image("same", image_index)

    def test_checker_reports_all_supported_issue_types(self) -> None:
        dataset = self.make_dataset()
        valid = self.write_image(dataset.image_dir / "valid.jpg", 10)
        duplicate = dataset.image_dir / "duplicate.jpg"
        shutil.copy2(valid, duplicate)
        missing_label = self.write_image(dataset.image_dir / "missing.jpg", 30)
        bad = dataset.image_dir / "bad.jpg"
        bad.write_bytes(b"not an image")
        self.write_image(dataset.image_dir / "empty.jpg", 50)
        self.write_image(dataset.image_dir / "invalid.jpg", 60)
        ambiguous_jpg = self.write_image(dataset.image_dir / "ambiguous.jpg", 70)
        ambiguous_png = self.write_image(dataset.image_dir / "ambiguous.png", 80)

        valid_label = self.write_label(
            dataset.label_dir / "valid.txt",
            "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",
        )
        self.write_label(dataset.label_dir / "duplicate.txt")
        self.write_label(dataset.label_dir / "bad.txt")
        empty_label = self.write_label(dataset.label_dir / "empty.txt", "")
        invalid_label = self.write_label(
            dataset.label_dir / "invalid.txt",
            "0 0.5 0.5 0.2\n",
        )
        self.write_label(dataset.label_dir / "ambiguous.txt")
        orphan_label = self.write_label(dataset.label_dir / "orphan.txt")

        with patch(
                "cvkit.core.dataset.check.tqdm",
                side_effect=lambda iterable, **_: iterable,
        ):
            result = DatasetChecker(dataset).check()

        self.assertEqual(result.bad_images, [bad])
        self.assertEqual(result.missing_labels, [missing_label])
        self.assertEqual(result.empty_labels, [empty_label])
        self.assertEqual(result.orphan_labels, [orphan_label])
        self.assertEqual(set(result.invalid_labels), {invalid_label})
        self.assertIn("invalid", result.invalid_labels[invalid_label].lower())
        self.assertEqual(set(result.ambiguous_images["ambiguous"]), {ambiguous_jpg, ambiguous_png})

        self.assertEqual(len(result.repeat_labels), 1)
        repeat_label = result.repeat_labels[0]
        self.assertEqual(repeat_label.label_file, valid_label)
        self.assertEqual((repeat_label.first_index, repeat_label.second_index), (1, 2))
        self.assertAlmostEqual(repeat_label.iou, 1.0)

        self.assertEqual(len(result.repeat_images), 1)
        self.assertEqual(set(result.repeat_images[0]), {valid, duplicate})

    def test_cleaner_dry_run_does_not_modify_files(self) -> None:
        dataset = self.make_dataset()
        bad = dataset.image_dir / "bad.jpg"
        bad.write_bytes(b"bad")
        empty = self.write_label(dataset.label_dir / "empty.txt", "")
        orphan = self.write_label(dataset.label_dir / "orphan.txt")
        result = DatasetCheckResult(
            bad_images=[bad],
            empty_labels=[empty],
            orphan_labels=[empty, orphan],
        )

        plan = DatasetCleaner(dataset).clean(result, dry_run=True)

        self.assertEqual(len(plan), 3)
        self.assertTrue(any(item.startswith("MOVE:") for item in plan))
        self.assertEqual(sum(item.startswith("DELETE LABEL:") for item in plan), 2)
        self.assertTrue(bad.exists())
        self.assertTrue(empty.exists())
        self.assertTrue(orphan.exists())
        self.assertFalse((dataset.data_dir / "images_bad").exists())

    def test_cleaner_moves_bad_images_and_removes_labels(self) -> None:
        dataset = self.make_dataset()
        bad = dataset.image_dir / "bad.jpg"
        bad.write_bytes(b"bad")
        empty = self.write_label(dataset.label_dir / "empty.txt", "")
        orphan = self.write_label(dataset.label_dir / "orphan.txt")
        result = DatasetCheckResult(
            bad_images=[bad],
            empty_labels=[empty],
            orphan_labels=[empty, orphan],
        )

        DatasetCleaner(dataset).clean(result, dry_run=False)

        self.assertFalse(bad.exists())
        self.assertEqual((dataset.data_dir / "images_bad" / bad.name).read_bytes(), b"bad")
        self.assertFalse(empty.exists())
        self.assertFalse(orphan.exists())

    def test_statistics_counts_files_classes_and_images_per_class(self) -> None:
        dataset = self.make_dataset()
        for index, value in enumerate((10, 20, 30)):
            self.write_image(dataset.image_dir / f"image_{index}.jpg", value)

        self.write_label(
            dataset.label_dir / "image_0.txt",
            "0 0.2 0.2 0.1 0.1\n0 0.7 0.7 0.1 0.1\n1 0.5 0.5 0.2 0.2\n",
        )
        self.write_label(dataset.label_dir / "image_1.txt", "1 0.5 0.5 0.2 0.2\n")
        self.write_label(dataset.label_dir / "image_2.txt", "")

        statistics = DatasetStatistics(dataset)

        self.assertEqual(statistics.count_images(), 3)
        self.assertEqual(statistics.count_labels(), 3)
        self.assertEqual(statistics.count_classes(), Counter({0: 2, 1: 2}))
        self.assertEqual(statistics.images_per_class(), Counter({1: 2, 0: 1}))

    def test_splitter_creates_matching_train_and_validation_pairs(self) -> None:
        dataset = self.make_dataset()
        for index, value in enumerate((10, 20, 30, 40)):
            extension = ".png" if index % 2 else ".jpg"
            self.write_image(dataset.image_dir / f"image_{index}{extension}", value)
            self.write_label(dataset.label_dir / f"image_{index}.txt")

        DatasetSplitter(dataset).split(ratio=0.5, seed=7, max_workers=2)

        split_dir = dataset.data_dir / "split"
        train_images = list((split_dir / "train" / "images").iterdir())
        train_labels = list((split_dir / "train" / "labels").iterdir())
        val_images = list((split_dir / "val" / "images").iterdir())
        val_labels = list((split_dir / "val" / "labels").iterdir())

        self.assertEqual((len(train_images), len(train_labels)), (2, 2))
        self.assertEqual((len(val_images), len(val_labels)), (2, 2))
        self.assertEqual({path.stem for path in train_images}, {path.stem for path in train_labels})
        self.assertEqual({path.stem for path in val_images}, {path.stem for path in val_labels})
        self.assertEqual(
            {path.stem for path in train_images + val_images},
            {f"image_{index}" for index in range(4)},
        )
        self.assertEqual(len(list(dataset.iter_images())), 4)
        self.assertEqual(len(list(dataset.iter_labels())), 4)

        with self.assertRaises(FileExistsError):
            DatasetSplitter(dataset).split()

    def test_splitter_rejects_invalid_ratios(self) -> None:
        dataset = self.make_dataset()
        splitter = DatasetSplitter(dataset)

        for ratio in (0, 1, -0.1, 1.1):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    splitter.split(ratio=ratio)

    def test_splitter_rejects_dataset_without_labels(self) -> None:
        dataset = self.make_dataset()
        self.write_image(dataset.image_dir / "image.jpg", 10)

        with self.assertRaises(FileNotFoundError):
            DatasetSplitter(dataset).split()
        self.assertFalse((dataset.data_dir / "split").exists())

    def test_splitter_rejects_label_without_image(self) -> None:
        dataset = self.make_dataset()
        self.write_label(dataset.label_dir / "missing.txt")

        with self.assertRaisesRegex(FileNotFoundError, "Image not found"):
            DatasetSplitter(dataset).split()
        self.assertFalse((dataset.data_dir / "split").exists())

    def test_splitter_rejects_ambiguous_image_stem(self) -> None:
        dataset = self.make_dataset()
        self.write_image(dataset.image_dir / "same.jpg", 10)
        self.write_image(dataset.image_dir / "same.png", 20)
        self.write_label(dataset.label_dir / "same.txt")

        with self.assertRaisesRegex(ValueError, "同名图片存在歧义"):
            DatasetSplitter(dataset).split()
        self.assertFalse((dataset.data_dir / "split").exists())

    def test_group_checker_finds_duplicates_across_datasets(self) -> None:
        first = self.make_dataset("first")
        second = self.make_dataset("second")
        source = self.write_image(first.image_dir / "source.jpg", 10)
        copied = second.image_dir / "copied.jpg"
        shutil.copy2(source, copied)
        self.write_image(first.image_dir / "unique_first.jpg", 20)
        self.write_image(second.image_dir / "unique_second.jpg", 30)

        groups = DatasetGroupChecker([first, second]).find_repeat_images_group()

        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]), {str(source), str(copied)})

    def test_check_result_instances_do_not_share_mutable_defaults(self) -> None:
        first = DatasetCheckResult()
        second = DatasetCheckResult()
        first.bad_images.append(Path("bad.jpg"))
        first.invalid_labels[Path("bad.txt")] = "invalid"

        self.assertEqual(second.bad_images, [])
        self.assertEqual(second.invalid_labels, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
