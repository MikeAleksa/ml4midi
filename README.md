# Overview

This project was undertaken as a research project for CISC 4900 during the Spring 2020 semester at Brooklyn College.

The goal of this project was to use machine learning algorithms to programmatically generate piano music in various compositional styles, by learning from MIDI data of classical compositions.

Examples of output from the project can be heard on [soundcloud](https://soundcloud.com/ml4midi/sets/ml-generated-sequences).

# Quickstart

Example Jupyter Notebooks are provided which demonstrate simple pipelines to train and generate new MIDI compositions from a dataset of MIDI files. No previous knowledge of TensorFlow or machine learning is necessary to use the model with default settings. It is recommended to use a GPU for training, or to use the provided checkpoints of pre-trained models.

The MusicModel class provides an interface for experimenting with the architecture and hyperparameters of the model. This class inherits from tensorflow.keras.Sequential, and implements an architecture of Embedding -> LSTM -> Dense layers, including batch normalization and dropout for all layers.

The serializers module provides an abstract base class, MidiSerializer. This project used a MIDI serialization module based on discrete time steps for "wait events", but those looking to experiment can implement a different serialization class as long as it inherits from MidiSerializer.

# Requirements

* TensorFlow 2
* Python3
* GPU

# Project Details

To create a supervised machine learning pipeline that can learn from a dataset of compositions I first had to collect and preprocess the datasets to be sequences which the model could use for training. My datasets were in the form of MIDI files. I chose to serialize the MIDI files into sequences of single events represented by integers, where each event is a MIDI note-on event, MIDI note-off event, or a wait event. No MIDI velocity information was used. Wait events represent discrete time steps in milliseconds to avoid programming any musical knowledge or information, such as meter and note divisions, into the model.

These sequences are split into smaller windows of events and an associated label, and used as training data. The label for a given sequence is the following event from the composition. The model is trained to predict the next event, given a sequence of events, by minimizing the categorical-crossentropy loss between the predicted next event and the given label. Essentially, the prediction of each event, given a sequence of events, is a single-label classification problem where the model tries to answer what the most-likely next event is. Depending on size and variety in each training dataset, transposition of compositions into different keys can be used to augment the dataset.

The model is built on the TensorFlow library, and its Keras API, and consists of a series of embedding, LSTM, and dense neural network layers. The embedding layer is used to turn integers into dense vectors which represent the association of different events. Sequences of these vectors are then fed to recurrent neural network (RNN) layers of LSTM memory cells, which are cells with an input gate, output gate, and forget gate, allowing the network to remember values over time. This makes them useful for sequence and time-series predictions. Finally, these LSTM cells feed to dense neural network layers, which are used to calculate the probability density of the next event over all possible events.

Generating new musical sequences from a trained model is achieved by sampling from the model. This is done by supplying the model with a sequence to 'seed' the prediction of the first event. The model calculates the probability distribution of the next event over the sample space of all possible events. This probability distribution is used to randomly select the next event. The chosen event is then added to the sequence and used to make another prediction. Each sampled event is subsequently added to the sequence and used to make the next prediction. Sequence history is truncated to include only a window of most-recent events for predicting and randomly selecting the next event. Random sampling using probability distributions allows the model to produce different sequences given the same seed. The amount of randomness can be controlled using a 'temperature' control, allowing more or less-likely sequences to be produced where the most-likely sequence to be produced would be made by taking the most probable event (without any randomness) each time.
