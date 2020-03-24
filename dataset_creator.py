class DatasetCreator:

    def __init__(self, sequences, window_size: int = 128, shift_size: int = 1):
        """
        :param sequences: event sequences
        :param window_size: size of windows to make from event sequences
        :param shift_size: shift size between windows
        """
        self.sequences = sequences
        self.window_size = window_size
        self.shift_size = shift_size

    def window(self):
        """
        window and split event sequences into lists of windows and labels
        :return: lists containing windows from the event sequence, and labels (next event) for each window
        """
        windows = []
        labels = []
        for sequence in self.sequences:
            for i in range(0, len(sequence) - self.window_size, self.shift_size):
                windows.append(sequence[i:i + self.window_size])
                labels.append(sequence[i + self.window_size])
        return windows, labels
