from abc import ABC, abstractmethod
from pathlib import Path

import mido


class MidiConverter(ABC):
    """
    Abstract base class for MIDI converter.
    """

    @abstractmethod
    def __init__(self, n_classes: int):
        """
        Set up any variables needed for MIDI conversion.
        :param n_classes: the number of event classes used in conversion scheme
        """
        self.n_classes = n_classes

    @abstractmethod
    def convert_file(self, path) -> list:
        """
        Convert a single file from MIDI to an event sequence.
        :param path: the path to the MIDI file
        :return: an event sequence
        """
        pass

    @abstractmethod
    def convert_folder(self, path) -> list:
        """
        Convert a folder of MIDI files into a list of event sequences.
        :param path: the path to the folder of MIDI files
        :return: a list of event sequences
        """
        if not isinstance(path, Path):
            path = Path(path)
        files = path.glob('*.mid')
        sequences = []
        for file in files:
            sequences.append(self.convert_file(file))
        return sequences

    @abstractmethod
    def convert_to_file(self, seq, path, filename):
        """
        Convert an event sequence into a MIDI file.
        :param seq: an event sequence
        :param path: the output path for the MIDI file
        :param filename: the output filename for the MIDI file
        """
        pass


class DiscreteTimeMidiConverter(MidiConverter):
    """
    A MIDI converter that quantizes events to discrete time steps.
    """

    def __init__(self, samples: int = 500, bpm: int = 120, wait_classes: int = 100):
        """
        Set up variables to use for quantization and conversion of events between BPM and discrete time steps
        :param samples: the number of discrete samples per second, used to quantize event - defaults to 2ms steps
        :param bpm: the bpm for any output midi files
        :param wait_classes: the number of classes representing 'wait time' events
        """
        super().__init__(n_classes=256 + wait_classes)
        self.wait_classes = wait_classes
        self.samples = samples
        self.BPM = bpm
        self.TPB = int(self.samples * 60 * (1 / self.BPM))

    def convert_file(self, path) -> list:
        """
        Convert a single file from MIDI to an event sequence, quantized into discrete time steps.
        :param path: the path to the MIDI file
        :return: an event sequence
        """
        if not isinstance(path, Path):
            path = Path(path)
        midi = mido.MidiFile(path)
        seq = []

        chord = set()
        time = 0.0
        for msg in midi:
            # capture time delta for all events, even if not note on/off
            time += msg.time
            if msg.type in ['note_on', 'note_off']:
                if time != 0.0:
                    # append the current chord in sorted order
                    seq = seq + sorted(chord, reverse=True)
                    chord.clear()

                    # append wait events
                    max_wait_event = self.wait_classes - 1
                    scaled_time = int(round(time * self.samples)) - 1
                    while scaled_time > max_wait_event:
                        seq.append(max_wait_event + 256)
                        scaled_time = scaled_time - self.wait_classes
                    if scaled_time >= 0:
                        seq.append(scaled_time + 256)
                    time = 0.0

                    # append new note to chord
                    if msg.type == 'note_on' and msg.velocity != 0:
                        chord.add(msg.note)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        chord.add(msg.note + 128)
                else:
                    if msg.type == 'note_on' and msg.velocity != 0:
                        chord.add(msg.note)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        chord.add(msg.note + 128)
        if chord:
            seq = seq + sorted(chord, reverse=True)

        return seq

    def convert_folder(self, path) -> list:
        """
        Convert a folder of MIDI files into a list of event sequences.
        :param path: the path to the folder of MIDI files
        :return: a list of event sequences
        """
        return super().convert_folder(path)

    def convert_to_file(self, seq, path, filename):
        """
        Convert an event sequence into a MIDI file.
        :param seq: an event sequence
        :param path: the output path for the MIDI file
        :param filename: the output filename for the MIDI file
        """
        tempo = mido.bpm2tempo(self.BPM)
        midi = mido.MidiFile(ticks_per_beat=self.TPB)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        time_delta = 0

        if not isinstance(path, Path):
            path = Path(path)
        if not path.is_dir():
            path.mkdir()

        for event in seq:
            if event >= 256:
                time_delta += int(round(mido.second2tick(((event - 255) / self.samples), self.TPB, tempo)))
            elif event < 128:
                track.append(mido.Message('note_on', note=event, time=time_delta))
                time_delta = 0
            else:
                track.append(mido.Message('note_off', note=event - 128, time=time_delta))
                time_delta = 0
        midi.save(path / filename)
