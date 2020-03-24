from pathlib import Path

import mido


class MidiConverter:

    def __init__(self, samples: int = 500, bpm: int = 120):
        """
        :param samples: the number of samples/second, used in quantization - default quantizes to 2ms increments
        :param bpm: the bpm for any output midi files
        """
        self.samples = samples
        self.bpm = bpm
        self.tpb = int(self.samples * 60 * (1 / self.bpm))

    def sequence_from_midi_folder(self, path) -> list:
        """
        convert a folder of MIDI files into a list of event sequences
        :param path: path to folder of files to convert
        :return: a list of event sequences
        """
        if not isinstance(path, Path):
            path = Path(path)
        files = path.glob('*.mid')
        sequences = []
        for file in files:
            sequences.append(self.sequence_from_midi_file(file))
        return sequences

    def sequence_from_midi_file(self, path) -> list:
        """
        convert a single MIDI file into an event sequence
        :param path: path to file to convert
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
                    scaled_time = int(round(time * self.samples)) - 1
                    while scaled_time > 127:
                        seq.append(127 + 256)
                        scaled_time = scaled_time - 128
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

    def sequence_to_midi_file(self, seq, path, filename):
        """
        convert an event sequence into a midi file
        :param seq: event sequence to convert to midi
        :param path: path to output midi file
        :param filename: filename for midi file
        :return:
        """
        tempo = mido.bpm2tempo(self.bpm)
        midi = mido.MidiFile(ticks_per_beat=self.tpb)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        time_delta = 0

        if not isinstance(path, Path):
            path = Path(path)
        if not path.is_dir():
            path.mkdir()

        for event in seq:
            if event >= 256:
                time_delta += int(round(mido.second2tick(((event - 255) / self.samples), self.tpb, tempo)))
            elif event < 128:
                track.append(mido.Message('note_on', note=event, time=time_delta))
                time_delta = 0
            else:
                track.append(mido.Message('note_off', note=event - 128, time=time_delta))
                time_delta = 0

        midi.save(path / filename)
