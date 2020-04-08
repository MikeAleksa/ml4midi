from abc import ABC, abstractmethod
from pathlib import Path


class MidiSerializer(ABC):
    """
    Abstract base class for MIDI serialization.
    """

    @abstractmethod
    def __init__(self, n_classes: int):
        """
        Set up any variables needed for MIDI serialization.
        :param n_classes: the number of event classes used in serialization scheme
        """
        self.n_classes = n_classes

    @abstractmethod
    def serialize(self, path) -> list:
        """
        Serialize a single file from MIDI to an event sequence.
        :param path: the path to the MIDI file
        :return: an event sequence
        """
        pass

    @abstractmethod
    def serialize_folder(self, path) -> list:
        """
        Serialize a folder of MIDI files into a list of event sequences.
        :param path: the path to the folder of MIDI files
        :return: a list of event sequences
        """
        if not isinstance(path, Path):
            path = Path(path)
        files = path.glob('*.mid')
        sequences = []
        for file in files:
            sequences.append(self.serialize(file))
        return sequences

    @abstractmethod
    def deserialize(self, seq, path, filename):
        """
        De-serialize an event sequence into a MIDI file.
        :param seq: an event sequence
        :param path: the output path for the MIDI file
        :param filename: the output filename for the MIDI file
        """
        pass
