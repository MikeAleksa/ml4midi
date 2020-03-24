import mido

from pathlib import Path


class MidiConverter:

    def __init__(self, path: str, samples: int = 100):
        """
        :param samples: the number of samples/second, used in quantization - default of 100 quantizes to 10ms increments
        """
        self.samples = samples

    def batch_midi_to_sequence(self, path) -> list:
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
            sequences.append(self.midi_to_sequence(file))
        return sequences

    def midi_to_sequence(self, path) -> list:
        """
        convert a single MIDI file into an event sequence
        :param path: path to file to convert
        :return: an event sequence
        """
        if not isinstance(path, Path):
            path = Path(path)
        midi = mido.MidiFile(path)
        seq = []

        # iterate over note_on and note_off messages
        chord = set()
        time = 0.0
        for msg in midi:
            # capture time delta for all events, even if not note on/off
            time += msg.time
            if msg.type == 'note_on' or msg.type == 'note_off':
                # if time is zero the note is part of a chord - append to chord
                if msg.time == 0:
                    if msg.type == 'note_on':
                        chord.add(msg.note)
                    else:
                        chord.add(msg.note + 128)
                else:
                    # append the current chord in sorted order
                    seq = seq + sorted(chord, reverse=True)
                    chord = set()

                    # append wait events
                    scaled_time = int(round(time * self.samples)) - 1
                    while scaled_time > 127:
                        seq.append(127 + 256)
                        scaled_time = scaled_time - 128
                    if scaled_time >= 0:
                        seq.append(scaled_time + 256)
                    time = 0.0

                    # append new note to chord
                    if msg.type == 'note_on':
                        chord.add(msg.note)
                    else:
                        chord.add(msg.note + 128)

        return seq

    def sequence_to_midi(self, seq, output_path, tpb: int = 960, bpm: int = 120):
        """
        convert an event sequence into a midi file
        :param seq: event sequence to convert to midi
        :param output_path: path to output midi file to
        :param tpb: ticks per beat for resulting midi file
        :param bpm: beats per minute for resulting midi file
        """
        tempo = mido.bpm2tempo(bpm)
        midi = mido.MidiFile(ticks_per_beat=tpb)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        time_delta = 0

        for event in seq:
            if event >= 256:
                time_delta += int(round(mido.second2tick(((event - 256 + 1) / time_scale), tpb, tempo)))
            elif event < 128:
                track.append(mido.Message('note_on', note=event, time=time_delta))
                time_delta = 0
            else:
                track.append(mido.Message('note_off', note=event - 128, time=time_delta))
                time_delta = 0

        midi.save(output_path)
