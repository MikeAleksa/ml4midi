class NoLabelException(Exception):
    pass


class DatasetCreator:

    def __init__(self, data):
        """
        :param data: data for the dataset
        """
        self.data = data
        self.labels = None

    def get_tf_dataset(self):
        pass

    def make_labels(self):
        pass

    def normalize(self, ):
        pass

    def transpose(self, semitones: int):
        pass

    def window(self, window_size: int = 128, shift_size: int = 1):
        """
        window and split event sequences into lists of windows and labels
        :param window_size: size of windows to make from event sequences
        :param shift_size: shift size between windows
        :return: lists containing windows from the event sequence, and labels (next event) for each window
        """
        windows = []
        labels = []
        for sequence in self.sequences:
            for i in range(0, len(sequence) - window_size, shift_size):
                windows.append(sequence[i:i + window_size])
                labels.append(sequence[i + window_size])
        return windows, labels

