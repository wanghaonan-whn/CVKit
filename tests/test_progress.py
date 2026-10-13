import io
import unittest
from unittest.mock import patch

from cvkit.core.dataset.progress import iter_progress


class TerminalStream(io.StringIO):
    def isatty(self):
        return True


class ProgressTestCase(unittest.TestCase):
    def test_terminal_counts_only_completed_items(self):
        stream = TerminalStream()
        with patch("sys.stderr", stream):
            progress = iter_progress(["a", "b"], step=1, label="检查图片", min_interval=0)
            self.assertEqual(next(progress), "a")
            self.assertNotIn("1 / 2", stream.getvalue())
            self.assertEqual(next(progress), "b")
            self.assertIn("1 / 2", stream.getvalue())
            self.assertNotIn("2 / 2", stream.getvalue())
            self.assertEqual(list(progress), [])
        self.assertIn("[1/6] 检查图片：2 / 2\r\n", stream.getvalue())

    def test_redirected_output_has_no_carriage_returns(self):
        stream = io.StringIO()
        with patch("sys.stderr", stream), patch("time.monotonic", return_value=0):
            self.assertEqual(list(iter_progress(list(range(1000)), step=2, label="检查标注")),
                             list(range(1000)))
        self.assertEqual(stream.getvalue().splitlines(),
                         ["[2/6] 检查标注：0 / 1000", "[2/6] 检查标注：1000 / 1000"])

    def test_non_terminal_reports_intermediate_progress_once_per_second(self):
        stream = io.StringIO()
        with patch("sys.stderr", stream), patch(
            "time.monotonic", side_effect=[0, 0.2, 1.0, 1.3, 2.1, 2.2]
        ):
            self.assertEqual(list(iter_progress([1, 2, 3, 4, 5], step=1, label="检查图片")),
                             [1, 2, 3, 4, 5])
        self.assertEqual(stream.getvalue().splitlines(), [
            "[1/6] 检查图片：0 / 5",
            "[1/6] 检查图片：2 / 5",
            "[1/6] 检查图片：4 / 5",
            "[1/6] 检查图片：5 / 5",
        ])

    def test_empty_stage_reports_zero(self):
        stream = io.StringIO()
        with patch("sys.stderr", stream):
            self.assertEqual(list(iter_progress([], step=3, label="检查标注")), [])
        self.assertEqual(stream.getvalue(), "[3/6] 检查标注：0 / 0\n")

    def test_interrupted_iteration_does_not_report_success(self):
        stream = TerminalStream()
        with patch("sys.stderr", stream):
            progress = iter_progress([1, 2], step=1, label="检查图片")
            next(progress)
            progress.close()
        self.assertNotIn("2 / 2", stream.getvalue())
        self.assertTrue(stream.getvalue().endswith("\n"))
