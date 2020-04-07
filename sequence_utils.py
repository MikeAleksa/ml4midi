"""
A collection of utilities for manipulating event sequences.
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


def transpose(sequences: list, down: int = 0, up: int = 0) -> list:
    """
    Transpose event sequences or labels by a range of semitones.
    :param sequences: list of event sequences
    :param down: range of transposition down, in semitones
    :param up: range of transposition up, in semitones
    :return: a list containing all event sequences and their transpositions
    """
    transposed_sequences = []
    for elem in sequences:
        for semitone in range(down, up+1):
            if isinstance(elem, list):
                transposed_sequences.append(transpose_sequence(elem, semitone))
            elif isinstance(elem, int):
                transposed_sequences.append(transpose_event(elem, semitone))
    return transposed_sequences


def transpose_sequence(sequence: list, semitones: int) -> list:
    """
    Transpose notes in a sequence of events.

    Transposition is limited to one octave up/down. Notes already in the highest/lowest octave will not be transposed.
    :param sequence: a sequence of events
    :param semitones: the number of semitones to transpose
    :return: a transposed event sequence
    """
    assert (-12 <= semitones <= 12)
    transposed_sequence = []
    for event in sequence:
        new_event = transpose_event(event, semitones)
        transposed_sequence.append(new_event)
    return transposed_sequence


def transpose_event(event: int, semitones: int) -> int:
    """
    Transpose a single event label.

    Transposition is limited to one octave up/down. Notes already in the highest/lowest octave will not be transposed.
    :param event: an integer representing an event class
    :param semitones: the number of semitones to transpose
    :return: a transposed event
    """
    assert (-12 <= semitones <= 12)
    new_event = event
    # transpose note-on events
    if 0 <= new_event <= 127:
        new_event = event + semitones
        if new_event > 127:
            new_event -= 12
        if new_event < 0:
            new_event += 12
    # transpose note-off events
    elif 128 <= new_event <= 255:
        new_event = event + semitones
        if new_event > 255:
            new_event -= 12
        if new_event < 128:
            new_event += 12
    return new_event