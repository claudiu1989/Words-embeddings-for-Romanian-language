
"""Popescu Claudiu, 2021.

Code for generating word2vec representations for Romanian language.

Usage:

python ./word2vec_generation_ro.py
"""

import time
import nltk
from gensim.models import Word2Vec
from nltk.stem.snowball import RomanianStemmer


def generate_embeddings(data_file, vector_size, preprocess, use_skipgram, use_hierarchical_softmax):
    """The main function.

    Args:
        data_file (str): Path to the corpus (a text file).
        vector_size (int): The size of the embeddings (e.g. 100, 200, 300).
        preprocess (bool): If true, apply preprocessing- covert to lowercase and apply stemming
        use_skipgram (bool): If true, use the Skip-gram model; otherwise Continuous Bag of Words Model (cbow) is used
        use_hierarchical_softmax (bool): If true, train by hierarchical softmax; otherwise negative sampling is used
    Returns:
        -
    """
    print('Start the training')
    start = time.time()
    with open(data_file, 'r', encoding='UTF-8') as data_file:
        lines = data_file.readlines()
        tokenized_lines = []
        if preprocess:
            rs = RomanianStemmer()
            for line in lines:
                words = nltk.word_tokenize(line)
                stemmed_words = [rs.stem(word.lower()) for word in words]
                tokenized_lines.append(stemmed_words)
        else:
            tokenized_lines = [nltk.word_tokenize(line) for line in lines]

        if use_skipgram:
            sg = 1
        else:
            sg = 0
        if use_hierarchical_softmax:
            hs = 1
        else:
            hs = 0

        model = Word2Vec(sentences=tokenized_lines, vector_size=vector_size, sg=sg, hs=hs, window=5, min_count=10, workers=4, negative=10)
        if preprocess:
            preprocess_part = '-preprocessed'
        else:
            preprocess_part = ''
        if use_skipgram:
            architecture_part = '-skipgram'
        else:
            architecture_part = '-cbow'
        if use_hierarchical_softmax:
            training_part = '-hierarchical_softmax'
        else:
            training_part = '-negative_sampling'
        end = time.time()
        processing_time = end-start
        print(f'Processing time: {processing_time}')
        with open('processing_time.txt', 'a') as time_file:
            time_file.write(f'Processing time for parameters vector_size:{vector_size},  preprocess:{preprocess}, use_skipgram:{use_skipgram}, use_hierarchical_softmax:{use_hierarchical_softmax}: {processing_time}\n')
        model.save(f"word2vec-ro-{vector_size}{preprocess_part}{architecture_part}{training_part}.model")


if __name__ == '__main__':
    # A Romanian language corpus (e.g. Wikipedia Ro dump)
    data_file = './wiki_2019_ro.txt'
    generate_embeddings(data_file, 100, False, False, False)
    generate_embeddings(data_file, 200, False, False, False)
    generate_embeddings(data_file, 300, False, False, False)
    generate_embeddings(data_file, 100, True, False, False)
    generate_embeddings(data_file, 200, True, False, False)
    generate_embeddings(data_file, 300, True, False, False)

    generate_embeddings(data_file, 100, False, True, False)
    generate_embeddings(data_file, 200, False, True, False)
    generate_embeddings(data_file, 300, False, True, False)
    generate_embeddings(data_file, 100, True, True, False)
    generate_embeddings(data_file, 200, True, True, False)
    generate_embeddings(data_file, 300, True, True, False)

    generate_embeddings(data_file, 100, False, False, True)
    generate_embeddings(data_file, 200, False, False, True)
    generate_embeddings(data_file, 300, False, False, True)
    generate_embeddings(data_file, 100, True, False, True)
    generate_embeddings(data_file, 200, True, False, True)
    generate_embeddings(data_file, 300, True, False, True)

    generate_embeddings(data_file, 100, False, True, True)
    generate_embeddings(data_file, 200, False, True, True)
    generate_embeddings(data_file, 300, False, True, True)
    generate_embeddings(data_file, 100, True, True, True)
    generate_embeddings(data_file, 200, True, True, True)
    generate_embeddings(data_file, 300, True, True, True)
