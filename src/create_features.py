import os
import pickle
import sys
import numpy as np
import yaml
from keras.preprocessing.text import Tokenizer
from utils import path


def featurize(words, num_of_features, vocab_size=10000 ):
    """Creates features by using Keras tokenizer to transform a list of tokens into a list of integers.

    Keyword arguments:
    words -- words to be encoded.
    num_of_features -- number of words to predict next word
    vocab_size -- limits the vocabulary size to save memory and computation time (default value 10.000).

    returns X and y (features and labels)
    """
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts([words])
    encoded = tokenizer.texts_to_sequences([words])[0]
    #print(encoded[:100])

    sequences = list()
    for i in range(num_of_features, len(encoded)):
        sequence = encoded[i - num_of_features:i + 1]
        sequences.append(sequence)
    print('Total Sequences: %d' % len(sequences))

    sequences = np.array(sequences)
    X, y = sequences[:, :num_of_features], sequences[:, num_of_features]

    print('Features:')
    print(X[:5, :])
    print('Labels:')
    print(y[:5])

    return X, y


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python create_features.py <tokens_file>')
        sys.exit(1)
    input_file_path = sys.argv[1]

    features_path = os.path.join('..', 'data', 'preprocessed', 'features.pickle')
    path.create_if_not_exists(features_path)

    labels_path = os.path.join('..', 'data', 'preprocessed', 'labels.pickle')
    path.create_if_not_exists(labels_path)

    params = yaml.safe_load(open('../params.yaml'))['create_features']
    vocab_size = params['vocab_size']
    num_of_features = params['num_of_features']

    with open(input_file_path, 'r') as text:
        input_text = text.read()
        features, labels = featurize(words=input_text, num_of_features=num_of_features, vocab_size=vocab_size)
        with open(features_path, 'wb') as features_file:
            pickle.dump(features, features_file, pickle.HIGHEST_PROTOCOL)
        with open(labels_path, 'wb') as labels_file:
            pickle.dump(labels, labels_file, pickle.HIGHEST_PROTOCOL)
        text.close()
