"""
A collection of utilities for tuning and using the model.
"""

import tensorflow as tf


def check_cuda_and_gpu() -> list:
    warnings = []

    if not tf.test.is_built_with_cuda():
        warnings.append('WARNING: TensorFlow not built with CUDA support.')
    if not tf.test.is_built_with_gpu_support():
        warnings.append('WARNING: TensorFlow not built with GPU support.')
    if not tf.config.list_physical_devices('GPU'):
        warnings.append('WARNING: No GPU Available.')
    if not warnings:
        warnings.append('CUDA and GPU Available...')

    return warnings
