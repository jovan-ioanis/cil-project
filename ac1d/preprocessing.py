"""
    Team OpinionatedAnalysts.

    Data loading and general marshaling:
        1. creating vocabulary, word2index and index2word pickles
        2. filtering tweets to be long up to conf.MAX_TWEET_LENGTH
        3. labels tweets and shuffles
        4. replacing words with UNK if they don't belong to dictionary
        5. padding tweets to conf.MAX_TWEET_LENGTH
"""
from __future__ import division
import os
import pickle
import tensorflow as tf
import numpy as np
import nltk
import itertools
import random
from configuration import Config as conf

class Dic:
    """
        Holds word <-> id associations.
    """

    def __init__(self):
        self.word2index = None
        self.index2word = None
        self.vocabulary = None

    def load_existing(self):
        if os.path.isfile(conf.VOCAB_PICKLED):
            print("Loading VOCABULARY pickle..")
            self.vocabulary = self.unpickle(conf.VOCAB_PICKLED)
            if os.path.isfile(conf.W2I_PICKLED) and os.path.isfile(conf.I2W_PICKLED):
                print("Loading WORD2INDEX pickle..")
                self.word2index = self.unpickle(conf.W2I_PICKLED)
                print("Loading INDEX2WORD pickle..")
                self.index2word = self.unpickle(conf.I2W_PICKLED)
            else:
                self.create_w2i_i2w()
        else:
            print("No vocabulary provided!")
            return

    def set_dataset_and_generate(self, tweets):
        word_freq = nltk.FreqDist(itertools.chain(*tweets))
        print("Number of unique words is {}".format(len(word_freq)))
        top_freqs = word_freq.most_common(conf.VOCABULARY_SIZE - 2)
        self.add_words(top_freqs)
        self.create_w2i_i2w()

    def create_w2i_i2w(self):
        """
            Creates word2index and index2word dictionaries including BOS, EOS, PAD and UNK
            Loads pickles if necessary
        """
        if self.vocabulary is None:
            if os.path.isfile(conf.VOCAB_PICKLED):
                print("Loading VOCABULARY pickle..")
                self.vocabulary = self.unpickle(conf.VOCAB_PICKLED)
            else:
                print("Words not provided for making word2index and index2word")
                return
        print("Generating INDEX2WORD..")
        self.index2word = dict(enumerate(self.vocabulary.keys()))
        print("Generating WORD2INDEX..")
        self.word2index = dict(zip(self.index2word.values(), self.index2word.keys()))
        self.dump_w2i_i2w()

    @staticmethod
    def make_dir():
        if not os.path.exists(conf.PICKLED_VARS):
            os.makedirs(conf.PICKLED_VARS)

    def dump_w2i_i2w(self):
        self.make_dir()
        print("Dumping WORD2INDEX pickle..")
        self.dump(conf.W2I_PICKLED, self.word2index)
        print("Dumping WORD2INDEX pickle..")
        self.dump(conf.I2W_PICKLED, self.index2word)

    def dump(self, filename, to_dump):
        self.make_dir()
        with open(filename, 'wb') as f:
            pickle.dump(to_dump, f)

    def unpickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def add_words(self, words):
        print("Generating VOCABULARY..")
        self.vocabulary = dict(words)
        # self.vocabulary[conf.BOS] = 1
        # self.vocabulary[conf.EOS] = 1
        self.vocabulary[conf.PAD] = 1
        self.vocabulary[conf.UNK] = 1
        print("Dumping VOCABULARY pickle..")
        self.dump(conf.VOCAB_PICKLED, self.vocabulary)


def load_file(filename):
    """
        Loads tweets from file given, and removes all tweets longer than conf.MAX_TWEET_LENGTH-2
    :param      filename:    file path
    :return:    filtered list of tweets, each represented as list of words
    """
    print("Reading data from {}".format(filename))
    f = tf.gfile.GFile(filename, "r")
    tweets = f.read().decode("utf-8").split("\n")
    return filter_tweets(tweets)


def filter_tweets(tweets):
    previous_length = len(tweets)
    tweets = [tweet.split() for tweet in tweets if len(tweet.split()) <= conf.MAX_TWEET_LENGTH]
    print('Number of tweets kept after filtering is {} out of {}'.format(len(tweets), previous_length))
    p = len(tweets) / previous_length * 100
    print('{}% of original data is kept.'.format(p))
    return tweets


def label_and_merge(positive_tweets, negative_tweets):
    """
        Labels are formatted as [n, p] where:
            1. n=1, p=0 if label is negative
            2. n=0, p=1 if label is positive
    """
    print('Labeling all tweets..')

    positive_indicator = np.ones(shape=len(positive_tweets), dtype=int)
    negative_indicator = np.zeros(shape=len(negative_tweets), dtype=int)

    positive_labels = np.zeros((len(positive_tweets), conf.num_classes))        # [num_tweets, 2]
    negative_labels = np.zeros((len(negative_tweets), conf.num_classes))        # [num_tweets, 2]
    positive_labels[np.arange(len(positive_tweets)), positive_indicator] = 1
    negative_labels[np.arange(len(negative_tweets)), negative_indicator] = 1

    pos = list(zip(positive_tweets, positive_labels))
    neg = list(zip(negative_tweets, negative_labels))

    merged = merge_tweets(pos, neg)
    return merged


def merge_tweets(pos, neg):
    print('Shuffling all tweets..')
    tw = []
    tw.extend(pos)
    tw.extend(neg)
    # random.shuffle(tw)
    return tw


def encode_tweets(all_tweets, dictionary):
    """
        Requires: list of tokenized tweets and built-up dictionary

        1. Replaces every word that is not in dictionary with UNK token
        2. Appends all tweets with PAD tokens up to conf.MAX_TWEET_LENGTH
        3. Every word is replaced with it's index from dictionary.word2index

        encoded_tweets and labels are of the same length, tweet on position i has label labels[i]
    """
    print('Encoding all tweets..')
    encoded_tweets_w_labels = []
    global_counter = 0
    unk_replacements_count = 0

    for tweet_tuple in all_tweets:
        tweet = []
        label = tweet_tuple[1]
        i = 0
        while i < conf.MAX_TWEET_LENGTH:
            if i < len(tweet_tuple[0]):
                word = tweet_tuple[0][i]
                if word in dictionary.word2index.keys():
                    tweet.append(dictionary.word2index[word])
                else:
                    tweet.append(dictionary.word2index[conf.UNK])
                    unk_replacements_count += 1
            else:
                tweet.append(dictionary.word2index[conf.PAD])
            i += 1
            global_counter += 1
        encoded_tweets_w_labels.append((label, tweet))

    p = unk_replacements_count / global_counter * 100
    print("{}% of words in dataset were replaced by UNK token".format(p))
    return encoded_tweets_w_labels


def load_dataset(full=False):
    if full:
        positive_tweets = load_file(conf.TRAIN_RAW_FULL % 'pos')
        negative_tweets = load_file(conf.TRAIN_RAW_FULL % 'neg')
    else:
        positive_tweets = load_file(conf.TRAIN_RAW % 'pos')
        negative_tweets = load_file(conf.TRAIN_RAW % 'neg')

    tweets = []
    for tweet in positive_tweets:
        tweets.append(tweet)
    for tweet in negative_tweets:
        tweets.append(tweet)

    dictionary = Dic()
    dictionary.load_existing()
    if dictionary.vocabulary is None:
        dictionary.set_dataset_and_generate(tweets)

    print("Vocabulary size: {}".format(len(dictionary.vocabulary)))

    all_tweets = label_and_merge(positive_tweets, negative_tweets)
    encoded_tweets_w_labels = encode_tweets(all_tweets, dictionary)

    return encoded_tweets_w_labels, dictionary


def load_test_data(dictionary):

    f = open('data/test_data.txt', 'r')

    tweets = []
    for i, line in enumerate(f):

        ID, tweet = line.split(',', 1)

        tw = []
        words = tweet.split()
        for word in words:
            if word in dictionary.vocabulary:
                tw.append(dictionary.word2index[word])
            else:
                tw.append(dictionary.word2index[conf.UNK])

        if len(tw) < conf.MAX_TWEET_LENGTH:
            tw.extend([dictionary.word2index[conf.PAD]] * (conf.MAX_TWEET_LENGTH - len(tw)))
        else:
            tw = tw[:conf.MAX_TWEET_LENGTH]

        tweets.append(tw)

    return tweets


def main():
    encoded_tweets, labels, dictionary = load_dataset()

    decoded_tweet = []
    print(encoded_tweets[0])
    for word in encoded_tweets[0]:
        decoded_tweet.append(dictionary.index2word[word])
    print(decoded_tweet)

if __name__ == "__main__":
    main()



