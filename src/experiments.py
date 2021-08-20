
"""Popescu Claudiu, 2021.

Basic experiments (analogy, retrieval of similar words, pair of words similarity computation) used to validate the embeddings.

Usage:

python ./experiments.py
"""

from gensim.models import Word2Vec

def analogy_experiment(model_file):  
    model = Word2Vec.load(model_file)

    result = model.wv.most_similar(positive=['roma', 'franța'], negative=['italia'], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=['regină', 'femeie'], negative=['bărbat'], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=['politician', 'necinstit'], negative=['cinstit'], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=['urât', 'alb'], negative=['frumos'], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=['frumos', 'negru'], negative=['urât'], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=['fizicianul', 'matematician'], negative=['fizician'], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=['lingvistul', 'informaticieni'], negative=['lingviști'], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=['praga', 'românia'], negative=['cehia'], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=[0.5*model.wv['alb'], 0.5*model.wv['negru']], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=[1.5*model.wv['rău']], negative=[0.5*model.wv['bun']], topn=10)  
    print(result)

    result = model.wv.most_similar(positive=[0.5*model.wv['rău'],0.5*model.wv['bun']], topn=10)  
    print(result)

def most_similar_words_experiment(model_file):  
    model = Word2Vec.load(model_file)
    result = model.wv.most_similar("casa", topn=10)
    print(result)
    result = model.wv.most_similar("fug", topn=10)
    print(result)
    result = model.wv.most_similar("metal", topn=10)
    print(result)
    result = model.wv.most_similar("zid", topn=10)
    print(result)
    
def similarity_between_two_words_experiment(model_file):  
    model = Word2Vec.load(model_file)
    result = model.wv.similarity('france', 'spain')
    print(result)
    result = model.wv.similarity('france', 'spain')
    print(result)
    result = model.wv.similarity('france', 'spain')
    print(result)
    result = model.wv.similarity('france', 'spain')
    print(result)
    result = model.wv.similarity('france', 'spain')
    print(result)

if __name__ == '__main__':
    # E.g.
    model_file = 'word2vec-ro-100-cbow-negative_sampling.model' # Be sure you have the model file in the cr dir
    print('\nSimilarity score')
    similarity_between_two_words_experiment(model_file)
    print('\nThe most similar words')
    most_similar_words_experiment(model_file)
    print('\nAnalogy')
    analogy_experiment(model_file)