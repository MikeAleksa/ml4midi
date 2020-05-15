"""
A collection of utilities useful while tuning and using the model.
"""
from midi2audio import FluidSynth
import tensorflow as tf

from pathlib import Path


def check_cuda_and_gpu() -> list:
    """
    Return warnings if a GPU device or CUDA is unavailable.
    """
    warnings = []

    if not tf.test.is_built_with_cuda():
        warnings.append('WARNING: TensorFlow not built with CUDA support.')
    if not tf.test.is_built_with_gpu_support():
        warnings.append('WARNING: TensorFlow not built with GPU support.')
    if not tf.config.list_physical_devices('GPU'):
        warnings.append('WARNING: No GPU Available.')
    if not warnings:
        warnings.append('CUDA and GPU Available.')

    return warnings

def synthesize_midi_files(str: folder) -> None:
    """
    Synthesize all MIDI files in a folder to WAV files using FluidSynth.
    """
    fs = FluidSynth()
    for file in Path(folder).glob('*.mid'):
        fs.midi_to_audio(str(file), str(file)[:-3] + 'wav')