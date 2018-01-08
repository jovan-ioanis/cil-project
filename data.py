"""
    Team OpinionatedAnalysts.

    Data loading and general marshaling.

    Instead of using the raw text files, we use some files of processed tweets.
    Each processed tweet is just a list of word IDs. A dictionary provides the
    association between word ID and word.

    This implies that we need to create these files first. This can be done
    simply by calling rewrite() once. After that the files are present and you
    don't need to call it on further runs.
"""

import os.path
import pickle
import random
from collections import Counter

from ac1d.configuration import Config as conf


class Dic:
    '''
    Holds word <-> id associations.
    '''

    def __init__(self, words=None):
        if words is not None:
            self.id_to_word = words
            for i, w in enumerate(words):
                self.word_to_id[w] = i
        else:
            self.word_to_id = {}
            self.id_to_word = []

    def add(self, w):
        '''
        Add a word to this dictionary.
        '''
        self.word_to_id[w] = len(self.id_to_word)
        self.id_to_word.append(w)

def process_tweet(s, dic):
    '''
    A processed tweet is a list of numbers, where each number refers to a word in
    the dictionary. This function constructs the processed tweet from a string.

    If a word of the tweet is not in the given dictionary, it is replaced with
    the string "<unk>". In addition, the processed tweet starts with <bos> and
    ends with <eos>.
    '''
    toks = s[:-1].split(' ')
    res = [dic.word_to_id['<bos>']]
    for t in toks:
        idx = dic.word_to_id.get(w, dic.word_to_id['<unk>'])
        res.append(idx)

    res.append(dic.word_to_id['<eos>'])
    return res

def recreate_tweet(tweet, dic):
    '''
    From a list of word IDs, recreate the original tweet
    '''
    def get(l, i, d):
        return (l[i] if i < len(l) else d)
    return ' '.join([get(dic.id_to_word, x, '<???>') for x in tweet])

def write_object(what, where):
    '''
    Pickle a single object in the given path
    '''
    with open(os.path.join(conf.DATA_ROOT, where), 'wb') as f:
        pickle.dump(what, f)

def read_object(where):
    '''
    Unpickle a single object from the given path
    '''
    with open(os.path.join(conf.DATA_ROOT, where), 'rb') as f:
        return pickle.load(f)


# TODO
# Get vocabulary counts
# churn it down as necessary
# then pass over the data again

def get_counts(file, counter):
    for ln in open(f, encoding='utf8'):
        for w in ln.split():
            counter[w] = counter.get(w, 0) + 1

def rewrite(vocab_limit):
    """
    Take all the samples in the raw text files and process them
    """
    train_pos_f = os.path.join(conf.DATA_ROOT, conf.TRAIN_RAW % 'pos')
    train_neg_f = os.path.join(conf.DATA_ROOT, conf.TRAIN_RAW % 'neg')
    test_f = os.path.join(conf.DATA_ROOT, conf.TEST_RAW)

    vocab_counts = Counter()
    get_counts(train_pos_f, vocab_counts)
    # get_counts(test_pos_f, vocab_counts)
    get_counts(test_f, vocab_counts)

    # Keep space for <unk>, <pad>, <eos>, <bos>
    top = vocab_counts.most_common(vocab_limit - 4)

    dic = Dic([x for x, _ in top])
    dic.add('<bos>')
    dic.add('<eos>')
    dic.add('<pad>')
    dic.add('<unk>')

    # List of tuples of (processed tweet, label) with label either 1 or -1
    train = []
    # List of tuples of (processed tweet, id) with the id of the test sample
    test = []

    dic = Dic()

    for ln in open(train_neg_f, encoding='utf8'):
        train.append((process_tweet(ln, dic), -1))

    for ln in open(train_pos_f, encoding='utf8'):
        train.append((process_tweet(ln, dic), 1))

    for ln in open(test_f, encoding='utf8'):
        # Separate the ID of the test case from the actual tweet.
        i = ln.find(',')
        test.append((process_tweet(ln[i + 1:], dic), int(ln[:i])))

    # Random order for training
    random.shuffle(train)

    # Dump as binaries
    write_object(train, TRAIN_PROCESSED % vocab_limit)
    write_object(test, TEST_PROCESSED % vocab_limit)
    write_object(dic, DIC % vocab_limit)


def load_train(vocab_limit):
    return read_object(TRAIN_PROCESSED % vocab_limit)

def load_test(vocab_limit):
    return read_object(TEST_PROCESSED % vocab_limit)

def load_dic(vocab_limit):
    return read_object(DIC % vocab_limit)

def generate_submission(Yhat, ids):
    f = open('submission.csv', 'w')
    f.write('ID,Prediction\n')
    for index, id in enumerate(ids):
        f.write('%d,%d\n' % (id, Yhat[index]))

    f.close()

