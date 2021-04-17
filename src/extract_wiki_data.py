"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
"""

import os
import sys

from gensim.corpora import WikiCorpus
from gensim.test.utils import datapath

from utils import path


def make_corpus(wiki_in_file, wiki_out_file):
    """Convert Wikipedia xml dump file to text corpus"""

    path_to_wiki_dump = datapath(wiki_in_file)

    with open(wiki_out_file, 'w') as output:
        wiki = WikiCorpus(path_to_wiki_dump)  # create word->word_id mapping
        i = 0
        for text in wiki.get_texts():
            output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
            i += 1
            if i % 10000 == 0:
                print('Processed ' + str(i) + ' articles')
        output.close()
        print('Processing complete!')


if __name__ == '__main__':

    if len(sys.argv) > 2:
        print('Usage: python exract_wiki_data.py <wikipedia_dump_file>')
        sys.exit(1)

    input_file = "enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2"
    if len(sys.argv) == 2:
        in_file = sys.argv[1]

    out_file = os.path.join('..', 'data', 'extracted', 'wiki_en.txt')
    path.create_if_not_exists(out_file)
    make_corpus(input_file, out_file)
