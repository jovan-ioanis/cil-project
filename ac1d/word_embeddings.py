"""
    Team OpinionatedAnalysts.

    Word Embeddings Training using word2vec technique from gensim package
"""

import gensim
import os
import numpy as np
import tensorflow as tf
import logging
from gensim import models
from configuration import Config as conf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def make_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)

def load_sentences(filepaths):
    """
        Loads sentences from various files.
        Expected input is: one tweet per line
            for test file: ID,tweet
    :param filepaths: list of paths to file to extract sentences from
    :return: list of tweets, each tweet represented as list of words
    """
    sentences = []

    for filepath in filepaths:
        print("\t\tLoading sentences from {}".format(filepath))
        f = open(filepath, 'r')
        for line in f:
            if filepath == conf.DATA_ROOT + '/' + conf.TEST_RAW:
                whole_line = line.strip().split(',')
                lineID = whole_line[0]
                tweet = []
                for i in range(1,len(whole_line)):
                    words = whole_line[i].strip().split(' ')
                    for word in words:
                        tweet.append(word.strip())
                sentences.append(tweet)
            else:
                tweet = line.strip().split(' ')
                sentences.append(tweet)

    return sentences


def train_embeddings():
    """
    Trains word embeddings using gensim package and dumps embeddings to disk
    :return: 
    """
    if os.path.isfile(conf.word2vec_filepath):
        return
    make_dir(conf.word2vec_ROOT)
    print("======== TRAINING WORD EMBEDDINGS ==========")
    print("WORD2VEC:\tembedding size = {}".format(conf.word2vec_embedding_size))
    print("WORD2VEC:\tmin word frequency = {}".format(conf.word2vec_min_frequency))
    print("WORD2VEC:\tnumber of workers = {}".format(conf.word2vec_num_workers))
    print("WORD2VEC:\tsave to path {}".format(conf.word2vec_filepath))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(load_sentences(conf.word2vec_training_paths),
                                   size=conf.word2vec_embedding_size,
                                   min_count=conf.word2vec_min_frequency,
                                   workers=conf.word2vec_num_workers)
    model.save(conf.word2vec_filepath)


def evaluate(path):
    print("Loading word2vec model from {}".format(path))
    model = gensim.models.Word2Vec.load(path)
    print('Vocabulary size is: {}'.format(len(model.wv.vocab)))
    print(model.most_similar(
        positive=['woman', 'king'], negative=['man'], topn=5))
    print(model.most_similar(
        positive=['girl', 'man'], negative=['boy'], topn=5))
    print(model.most_similar(
        positive=['head', 'eye', 'lips', 'nose'], negative=['toes'], topn=5))
    print(model.most_similar(
        positive=['pain'], topn=10))
    print(model.most_similar(
        positive=['pain'], negative=['happy'], topn=10))

    x = model[model.wv.vocab]
    # pca = PCA(n_components=2)
    # x_pca = pca.fit_transform(x[:5000,:])

    tsne = TSNE(n_components=2, random_state=0)
    x_tsne = tsne.fit_transform(x[:2000, :])

    # plt.scatter(x_pca[:, 0], x_pca[:, 1])
    # for label, x, y in zip(model.wv.vocab, x_pca[:, 0], x_pca[:, 1]):
    #     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    # plt.show()

    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    for label, x, y in zip(model.wv.vocab, x_tsne[:, 0], x_tsne[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def main():
    train_embeddings()
    evaluate(conf.word2vec_filepath)

if __name__ == "__main__":
    main()