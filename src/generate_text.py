import os
import random
import sys

import yaml
from keras import models
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from utils import path


def get_tokenizer(input_file_path, vocab_size):

    with open(input_file_path, 'r') as text:
        input_text = text.read()
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts([input_text])
        text.close()
    return tokenizer


def generate_seq(model, tokenizer, seed_text, n_words, vocab_size, num_of_features):
    textlist = seed_text.split()
    in_text, result = textlist[-5:], seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=num_of_features, padding='pre')

        probs = model.predict(encoded)
        yhat = random.choices(range(0, vocab_size), weights=probs[0], k=1)[0]
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text, result = out_word, result + ' ' + out_word
        print(out_word, end=' ')
    return result


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: python generate_text.py <path_to_model> <header_to_generate_text>\n')
        print('Usage: python generate_text.py "../data/trained" "The big breakthroughs in AI will be about language"')
        sys.exit(1)

    path_to_model = sys.argv[1]
    seed_text = sys.argv[2]

    params = yaml.safe_load(open('../params.yaml'))
    vocab_size = params['create_features']['vocab_size']
    feature_size = params['create_features']['num_of_features']
    text_size = params['generate_text']['text_size']

    model = models.load_model(path_to_model)
    tokenizer = get_tokenizer(os.path.join('..', 'data', 'preprocessed', 'tokens.txt'), vocab_size)
    generated_text = generate_seq(model, tokenizer, seed_text, text_size, vocab_size, feature_size)
    output_path = os.path.join('..', 'data', 'result', 'generated.txt')
    path.create_if_not_exists(output_path)
    with open(output_path, 'w') as output:
        output.write(generated_text)
