from pathlib import Path
from time import localtime, strftime

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class MusicModel:
    """
    LSTM-based model for music generation.
    """

    @property
    def summary(self):
        return self.model.summary

    def __init__(self,
                 n_classes: int,
                 embed_dims: int,
                 rnn_size: int,
                 rnn_layers: int,
                 dense_size: int,
                 dense_layers: int,
                 dropout_rate: float,
                 batch_norm: bool = True,
                 init_lr: float = 0.001,
                 ckpt_dir: str = './training_checkpoints',
                 log_dir: str = './logs'):
        """
        Initialize parameters, build and compile model.
        :param n_classes: the number of classes for the model to learn/predict
        :param embed_dims: embedding layer dimensions
        :param rnn_size: lstm layer units
        :param rnn_layers: number of (lstm, dropout, batch_normalization) layers
        :param dense_size: dense layer units
        :param dense_layers: number of dense layers
        :param dropout_rate: lstm-layer dropout rate
        :param batch_norm: a boolean value indicating whether or not to use batch normalization layers
        :param init_lr: the initial learning rate of the Adam optimizer
        :param ckpt_dir: directory to save checkpoints
        :param log_dir: directory to save tensorboard logs
        """
        self.n_classes = n_classes
        self.embed_dims = embed_dims
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.dense_size = dense_size
        self.dense_layers = dense_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.ckpt_path = str(Path(ckpt_dir) / 'ckpt_{epoch}')
        self.log_dir = str(Path(log_dir) / Path(strftime("%Y-%m-%d-%H%M", localtime())))
        self.callbacks = self.__define_callbacks()
        self.model = self.__build_model()
        self.history = None

        def loss(labels, logits):
            return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        # TODO: learning rate scheduling
        optimizer = keras.optimizers.Adam(learning_rate=init_lr)

        self.model.compile(loss=loss, optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

    def __define_callbacks(self) -> list:
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=self.ckpt_path,
                                                        save_weights_only=True)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                           histogram_freq=1,
                                                           profile_batch='2,5')
        return [ckpt_callback, tensorboard_callback]

    def __build_model(self) -> keras.Sequential:
        """
        Build and compile model.
        :return: a keras sequential model
        """
        model = keras.Sequential()

        # embedding layer
        model.add(keras.layers.Embedding(self.n_classes, self.embed_dims, batch_input_shape=[None, None]))

        # lstm layers
        for _ in range(self.rnn_layers - 1):
            model.add(keras.layers.LSTM(self.rnn_size, return_sequences=True))
            model.add(keras.layers.Dropout(self.dropout_rate))
            if self.batch_norm:
                model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LSTM(self.rnn_size))
        model.add(keras.layers.Dropout(self.dropout_rate))
        if self.batch_norm:
            model.add(keras.layers.BatchNormalization())

        # dense layers
        for _ in range(self.dense_layers - 1):
            model.add(keras.layers.Dense(units=self.dense_size, activation='relu'))
            if self.batch_norm:
                model.add(keras.layers.BatchNormalization())
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
        :param data: a dataset of event sequences - can be a tf.data.Dataset or tuple of (examples, labels)
        :param val_data: a dataset of validation data
        :param epochs: the number of epochs to train
        :param batch_size: the size of batch to train on (not used with tf.data.Dataset)
        :param verbose: verbosity
        :return: an object containing data about
        """
        if not isinstance(data, tf.data.Dataset):
            x, y = data
            self.history = self.model.fit(x,
                                          y,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          validation_data=val_data,
                                          verbose=verbose,
                                          callbacks=self.callbacks)
        else:
            self.history = self.model.fit(data,
                                          epochs=epochs,
                                          validation_data=val_data,
                                          verbose=verbose,
                                          callbacks=self.callbacks)
        return self.history

    def load_checkpoint(self, path: str, use_latest: bool = False):
        """
        Load weights from a checkpoint.
        :param path: the path to a checkpoint file
        :param use_latest: if true path should be a directory of checkpoints, otherwise path should be a checkpoint
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
