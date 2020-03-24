from unittest import TestCase

from dataset_creator import DatasetCreator


class TestDatasetCreator(TestCase):
    def setUp(self) -> None:
        seq1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        seq2 = [11, 12, 13, 14, 15, 16, 17]
        sequences = [seq1, seq2]
        self.dc = DatasetCreator(sequences, window_size=5)

    def test_window(self):
        windows, labels = self.dc.window()
        expected_windows = [[1, 2, 3, 4, 5],
                            [2, 3, 4, 5, 6],
                            [3, 4, 5, 6, 7],
                            [4, 5, 6, 7, 8],
                            [5, 6, 7, 8, 9],
                            [11, 12, 13, 14, 15],
                            [12, 13, 14, 15, 16]]
        expected_labels = [6, 7, 8, 9, 10, 16, 17]
        self.assertListEqual(expected_labels, labels)
        self.assertListEqual(expected_windows, windows)
