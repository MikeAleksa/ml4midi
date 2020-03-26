"""
A collection of utilities for manipulating datasets of note sequences.
"""


def window(sequences: list, window_size: int = 128, shift_size: int = 1):
    """
    Window and split event sequences into lists of windows and labels.
    :param sequences: list of event sequences
    :param window_size: size of windows to make from event sequences
    :param shift_size: shift size between windows
    :return: lists containing windows from the event sequence, and labels (next event) for each window
    """
    windows = []
    labels = []
    for sequence in sequences:
        for i in range(0, len(sequence) - window_size, shift_size):
            windows.append(sequence[i:i + window_size])
            labels.append(sequence[i + window_size])
    return windows, labels
