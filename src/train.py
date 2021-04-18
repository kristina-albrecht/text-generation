import os
import pickle
import sys
from random import randint

import yaml
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, save_model
from keras.layers import Dense, Embedding, LSTM
import numpy as np

from utils import path


def build_model(vocab_size, feature_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=feature_size))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(75))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    return model


def generate_batches(features, labels, batch_size, vocab_size, feature_size, corpus_size):
    X_batch = np.zeros((batch_size, feature_size))
    y_batch = np.zeros((batch_size, vocab_size))

    while True:
        for i in range(batch_size):
            index = randint(0, corpus_size - 1)
            X_batch[i] = features[index]
            y_batch[i] = to_categorical(labels[index], num_classes=vocab_size)
        yield X_batch, y_batch


def train(X, y, params, model_path):
    model = build_model(params['vocab_size'], params['feature_size'])
    batches = generate_batches(X, y, params['batch_size'], params['vocab_size'], params['feature_size'], params['corpus_size'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(batches, steps_per_epoch=params['steps_per_epoch'], epochs=params['epochs'], verbose=2, use_multiprocessing=True, workers=4)
    save_model(model, model_path) # save_format: Either 'tf' or 'h5', defaults to 'tf'


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python train.py <output_model_name>')
        sys.exit(1)
    model_name = sys.argv[1]

    with open('../data/preprocessed/features.pickle', 'rb') as features_pickle:
        features = pickle.load(features_pickle)
    with open('../data/preprocessed/labels.pickle', 'rb') as labels_pickle:
        labels = pickle.load(labels_pickle)

    yaml = yaml.safe_load(open('../params.yaml'))
    train_params = yaml['train']
    features_params = yaml['create_features']

    params = {'loss': 'categorical_crossentropy',
              'optimizer': 'adam',
              'vocab_size': features_params['vocab_size'],
              'feature_size': features_params['num_of_features'],
              'corpus_size': len(features),
              'batch_size': train_params['batch_size'],
              'steps_per_epoch': train_params['steps_per_epoch'],
              'epochs': train_params['epochs']}
    path_to_model = os.path.join('..', 'data', 'trained', model_name + '_' + str(params['epochs']) + '_' + str(params['steps_per_epoch']))
    path.create_if_not_exists(path_to_model)
    train(features, labels, params, path_to_model)
