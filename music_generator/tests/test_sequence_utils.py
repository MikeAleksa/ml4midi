from unittest import TestCase

from music_generator import sequence_utils


class TestSequenceUtils(TestCase):

    def test_window(self):
        """
        Check if windowing works on different length sequences.
        """
        sequences = [[0, 1, 2, 3, 4, 5], [10, 11, 12, 13]]
        windows, labels = sequence_utils.window(sequences, 3)
        self.assertListEqual(windows, [[0, 1, 2],
                                       [1, 2, 3],
                                       [2, 3, 4],
                                       [10, 11, 12]])
        self.assertListEqual(labels, [3, 4, 5, 13])

    def test_transpose(self):
        """
        Check if transposition to all keys works on a sequence.
        """
        sequences = [[10, 11, 12]]
        labels = [13]
        self.assertListEqual(sequence_utils.transpose(sequences, down=-6, up=5), [[4, 5, 6],
                                                                                  [5, 6, 7],
                                                                                  [6, 7, 8],
                                                                                  [7, 8, 9],
                                                                                  [8, 9, 10],
                                                                                  [9, 10, 11],
                                                                                  [10, 11, 12],
                                                                                  [11, 12, 13],
                                                                                  [12, 13, 14],
                                                                                  [13, 14, 15],
                                                                                  [14, 15, 16],
                                                                                  [15, 16, 17]])
        self.assertListEqual(sequence_utils.transpose(labels, down=-6, up=5),
                             [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

    def test_transpose_sequence(self):
        """
        Check if transposition up and down by a whole step works on a sequence with notes in the extreme octaves.
        """
        sequence = [0, 12, 115, 127, 256, 128, 140, 243, 255]
        self.assertListEqual([0, 12, 115, 127, 256, 128, 140, 243, 255],
                             sequence_utils.transpose_sequence(sequence, 0))
        self.assertListEqual([2, 14, 117, 117, 256, 130, 142, 245, 245],
                             sequence_utils.transpose_sequence(sequence, 2))
        self.assertListEqual([10, 10, 113, 125, 256, 138, 138, 241, 253],
                             sequence_utils.transpose_sequence(sequence, -2))

    def test_transpose_event(self):
        """
        Check transposition of a single event label
        """
        self.assertEqual(3, sequence_utils.transpose_event(1, 2))
        self.assertEqual(11, sequence_utils.transpose_event(0, -1))
        self.assertEqual(116, sequence_utils.transpose_event(127, 1))
        self.assertEqual(139, sequence_utils.transpose_event(128, -1))
        self.assertEqual(244, sequence_utils.transpose_event(255, 1))
        self.assertEqual(256, sequence_utils.transpose_event(256, 1))