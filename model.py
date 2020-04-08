from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class MusicModel(keras.Sequential):
    """
    LSTM-based model for music generation.
    """

    def __init__(self,
                 n_classes: int = 356,
                 embed_dims: int = 128,
                 rnn_size: int = 512,
                 rnn_layers: int = 3,
                 dense_size: int = 512,
                 dense_layers: int = 2,
                 dropout_rate: float = 0.1,
                 batch_norm: bool = True,
                 init_lr: float = 0.0003,
                 dense_activation: str = 'relu',
                 ckpt_dir: str = './training_checkpoints',
                 log_dir: str = './logs'):
        """
        Initialize parameters and build/compile model.
        :param n_classes: the number of classes for the model to learn/predict
        :param embed_dims: embedding layer dimensions
        :param rnn_size: lstm layer units
        :param rnn_layers: number of (lstm, dropout, batch_normalization) layers
        :param dense_size: dense layer units
        :param dense_layers: number of dense layers
        :param dropout_rate: lstm-layer dropout rate
        :param batch_norm: a boolean value indicating whether or not to use batch normalization layers
        :param init_lr: the initial learning rate of the Adam optimizer
        :param dense_activation: activation function for dense layers (excluding final output layer)
        :param ckpt_dir: directory to save checkpoints
        :param log_dir: directory to save tensorboard logs
        """
        super().__init__()
        self.n_classes = n_classes
        self.embed_dims = embed_dims
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.dense_size = dense_size
        self.dense_layers = dense_layers
        self.dense_activation = dense_activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.init_lr = init_lr
        self.ckpt_path = str(Path(ckpt_dir) / 'ckpt_{epoch}')
        self.log_dir = log_dir
        self.default_callbacks = self.__define_callbacks()

        def loss(labels, logits):
            return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        optimizer = keras.optimizers.Adam(learning_rate=init_lr, clipvalue=5.0)

        self.__build_model()
        self.compile(loss=loss, optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

    def __define_callbacks(self) -> list:
        """
        Define the default callbacks for MusicModel.
        :return: a list of callbacks
        """
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=self.ckpt_path,
                                                        save_weights_only=True)

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                           histogram_freq=1,
                                                           profile_batch='2,5')

        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                               factor=0.3,
                                                               patience=2,
                                                               min_lr=0.000001,
                                                               cooldown=5,
                                                               verbose=1)

        return [ckpt_callback,
                reduce_lr_callback,
                tensorboard_callback]

    def __build_model(self):
        """
        Build model using hyper-parameters of MusicModel.
        """
        # with embedding layer
        if self.embed_dims:
            self.add(keras.layers.Embedding(self.n_classes, self.embed_dims, batch_input_shape=[None, None]))
            # lstm layers
            for _ in range(self.rnn_layers - 1):
                self.add(keras.layers.LSTM(self.rnn_size, return_sequences=True))
                if self.dropout_rate:
                    self.add(keras.layers.Dropout(self.dropout_rate))
                if self.batch_norm:
                    self.add(keras.layers.BatchNormalization())
            self.add(keras.layers.LSTM(self.rnn_size))
            if self.dropout_rate:
                self.add(keras.layers.Dropout(self.dropout_rate))
            if self.batch_norm:
                self.add(keras.layers.BatchNormalization())

        # without embedding layer
        else:
            self.add(keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
            self.add(keras.layers.LSTM(self.rnn_size, return_sequences=True, input_shape=(None, 1)))
            if self.dropout_rate:
                self.add(keras.layers.Dropout(self.dropout_rate))
            if self.batch_norm:
                self.add(keras.layers.BatchNormalization())
            for _ in range(self.rnn_layers - 2):
                self.add(keras.layers.LSTM(self.rnn_size, return_sequences=True))
                if self.dropout_rate:
                    self.add(keras.layers.Dropout(self.dropout_rate))
                if self.batch_norm:
                    self.add(keras.layers.BatchNormalization())
            self.add(keras.layers.LSTM(self.rnn_size))
            if self.dropout_rate:
                self.add(keras.layers.Dropout(self.dropout_rate))
            if self.batch_norm:
                self.add(keras.layers.BatchNormalization())

        # dense layers
        for _ in range(self.dense_layers - 1):
            self.add(keras.layers.Dense(units=self.dense_size, activation=self.dense_activation))
            if self.dropout_rate:
                self.add(keras.layers.Dropout(self.dropout_rate))
            if self.batch_norm:
                self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.Dense(units=self.n_classes))

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):
        """
        Train the model for a fixed number of epochs, using keras.Sequential.fit() with default MusicModel callbacks.
        """
        super().fit(x=None,
                    y=None,
                    batch_size=None,
                    epochs=1,
                    verbose=1,
                    callbacks=self.default_callbacks + callbacks,
                    validation_split=0.,
                    validation_data=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None,
                    initial_epoch=0,
                    steps_per_epoch=None,
                    validation_steps=None,
                    validation_freq=1,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    **kwargs)

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
            self.load_weights(ckpt)
        except Exception as e:
            print(e)

    def finetune(self):
        # TODO: implement finetune
        pass

    def generate_sequence(self, length: int, seed_sequence: list, history: int = 100):
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
            next_sample_logits = self(sequence_history)
            next_sample = tf.random.categorical(next_sample_logits, 1)
            generated_sequence.append(int(next_sample))
        return generated_sequence[len(seed_sequence):]
