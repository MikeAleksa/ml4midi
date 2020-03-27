from pathlib import Path
from time import localtime, strftime

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class MusicModel:
    """
    LSTM-based model for music generation.
    """

    def __init__(self,
                 n_classes: int,
                 embed_dims: int,
                 rnn_size: int,
                 rnn_layers: int,
                 dense_size: int,
                 dense_layers: int,
                 dropout_rate: float,
                 ckpt_dir: str = './training_checkpoints'):
        """
        Initialize parameters, build and compile model.
        :param n_classes: the number of classes for the model to learn/predict
        :param embed_dims: embedding layer dimensions
        :param rnn_size: lstm layer units
        :param rnn_layers: number of (lstm, dropout, batch_normalization) layers
        :param dense_size: dense layer units
        :param dense_layers: number of dense layers
        :param dropout_rate: lstm-layer dropout rate
        :param ckpt_dir: directory to save checkpoints
        """
        self.n_classes = n_classes
        self.embed_dims = embed_dims
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.dense_size = dense_size
        self.dense_layers = dense_layers
        self.dropout_rate = dropout_rate
        self.ckpt_path = Path(ckpt_dir) / 'ckpt_{epoch}'
        self.log_dir = Path('./logs') / Path(strftime("%Y-%m-%d-%H%M", localtime()))
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(self.ckpt_path), save_weights_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(self.log_dir))
        self.callbacks = [ckpt_callback, tensorboard_callback]
        self.model = self.__build_model()

        # TODO: learning rate scheduling
        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        self.model.compile(loss=loss, optimizer='adam', metrics=[keras.metrics.SparseCategoricalAccuracy()])

    @property
    def summary(self):
        return self.model.summary

    def __build_model(self) -> keras.Sequential:
        """
        Build and compile model.
        :return: a keras sequential model
        """
        model = keras.Sequential()
        model.add(keras.layers.Embedding(self.n_classes, self.embed_dims, batch_input_shape=[None, None]))
        for _ in range(self.rnn_layers - 1):
            model.add(keras.layers.LSTM(self.rnn_size, return_sequences=True))
            model.add(keras.layers.Dropout(self.dropout_rate))
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LSTM(self.rnn_size))
        model.add(keras.layers.Dropout(self.dropout_rate))
        model.add(keras.layers.BatchNormalization())
        for _ in range(self.dense_layers - 1):
            model.add(keras.layers.Dense(units=self.dense_size))
        model.add(keras.layers.Dense(units=self.n_classes))
        return model

    def fit(self,
            data,
            val_data=None,
            epochs: int = 1,
            batch_size: int = None,
            verbose: int = 0) -> keras.callbacks.History:
        """
        Train model on dataset.
        :param data: a dataset of event sequences
        :param val_data: a dataset of validation data
        :param epochs: the number of epochs to train
        :param batch_size: the size of batch to train on (not used with tf.data.Dataset)
        :param verbose: verbosity
        :return: an object containing data about
        """
        if not isinstance(data, tf.data.Dataset):
            x, y = data
            history = self.model.fit(x,
                                     y,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     validation_data=val_data,
                                     verbose=verbose,
                                     callbacks=self.callbacks)
        else:
            history = self.model.fit(data,
                                     epochs=epochs,
                                     validation_data=val_data,
                                     verbose=verbose,
                                     callbacks=self.callbacks)
        return history

    def load_checkpoint(self, path: str, use_latest: bool = False):
        """
        Load weights from a checkpoint.
        :param path: the path to a checkpoint file
        :param use_latest: if true, path should be a directory of checkpoints, otherwise path should indicate a checkpoint
        """
        if use_latest:
            ckpt = tf.train.latest_checkpoint(path)
        else:
            ckpt = path
        try:
            self.model.load_weights(ckpt)
        except Exception as e:
            print(e)

    def finetune(self):
        # TODO: implement finetune
        pass

    def generate_sequence(self, length: int, seed_sequence: list, history: int = 128):
        """
        Generate an event sequence using a given event sequence as a seed.
        :param length: the number of events to generate
        :param seed_sequence: an event sequence to seed the prediction with - not included in final output
        :param history: the amount of events to consider for each prediction
        :return: an event sequence
        """
        # TODO: use temperature control
        generated_sequence = list()
        generated_sequence += seed_sequence
        for _ in range(length):
            sequence_history = np.array([generated_sequence[-history:]])
            next_sample_logits = self.model(sequence_history)
            next_sample = tf.random.categorical(next_sample_logits, 1)
            generated_sequence.append(int(next_sample))
        return generated_sequence[len(seed_sequence):]
