from pathlib import Path
from time import localtime, strftime

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
                 layers: int,
                 dropout_rate: float,
                 ckpt_dir: str = './training_checkpoints'):
        """
        Initialize parameters, build and compile model.
        :param n_classes: the number of classes for the model to learn/predict
        :param embed_dims: embedding layer dimensions
        :param rnn_size: lstm layer dimensions
        :param layers: number of (lstm, dropout, batch_normalization) layers
        :param dropout_rate: lstm-layer dropout rate
        :param ckpt_dir: directory to save checkpoints
        """
        self.n_classes = n_classes
        self.embed_dims = embed_dims
        self.rnn_size = rnn_size
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.ckpt_path = Path(ckpt_dir) / 'ckpt_{epoch}'
        self.log_dir = Path('/logs') / Path(strftime("%Y-%m-%d-%H%M", localtime()))
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.ckpt_path, save_weights_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        self.callbacks = [ckpt_callback, tensorboard_callback]
        self.model = self.__build_model()
        # TODO: learning rate scheduling
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __build_model(self) -> keras.Sequential:
        """
        Build and compile model.
        :return: a keras sequential model
        """
        model = keras.Sequential()
        model.add(keras.layers.Embedding(self.n_classes, self.embed_dims, batch_input_shape=[None, None]))
        for _ in range(self.layers - 1):
            model.add(keras.layers.LSTM(self.rnn_size, return_sequences=True))
            model.add(keras.layers.Dropout(self.dropout_rate))
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LSTM(self.rnn_size))
        model.add(keras.layers.Dropout(self.dropout_rate))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=self.n_classes, activation='softmax'))
        return model

    def fit(self,
            data: tf.data.Dataset,
            val_data: tf.data.Dataset,
            epochs: int,
            batch_size: int,
            verbose: int = 0) -> keras.callbacks.History:
        """
        Train model on dataset.
        :param data: a dataset of event sequences
        :param val_data: a dataset of validation data
        :param epochs: the number of epochs to train
        :param batch_size: the batch size to use during training
        :param verbose: verbosity
        :return: an object containing data about
        """
        history = self.model.fit(data,
                                 epochs=epochs,
                                 batch_size=batch_size,
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

    def generate_sequence(self):
        # TODO: implement generate_sequence
        pass
