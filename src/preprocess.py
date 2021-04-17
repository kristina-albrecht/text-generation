import os
import sys

import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils import path


def tokenize_words(text):
    # use nltk word tokenizer to produce tokens
    words = word_tokenize(text)
    print(words[0:100])

    # remove all stopwords
    filtered = filter(lambda token: token not in stopwords.words('english'), words)
    return " ".join(filtered)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python preprocess.py <text_file>')
        sys.exit(1)
    input_file_path = sys.argv[1]
    out_file_path = os.path.join('..', 'data', 'preprocessed', 'tokens.txt')
    path.create_if_not_exists(out_file_path)

    with open(input_file_path, 'r') as text:
        with open(out_file_path, 'w') as output:
            input_text = text.read()
            tokens = tokenize_words(input_text)
            output.write(tokens)
            text.close()
