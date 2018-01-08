import data
import gensim
import numpy as np

from sklearn.ensemble import RandomForestClassifier

WORD2VEC = '../data/gnews-vecs-neg300.bin'
LIMIT = 10000
VERBOSE = False
#LIMIT = None # NO LIMITS


def printv(s):
    if VERBOSE:
        print(s)

def tweet_embedding_by_average(tweet, dic, embeds):
    '''
    Generate a tweet embedding by taking the average of the individual embeddings.
    '''
    vecs = []

    for i in tweet:
        w = dic.id_to_word[i]
        try:
            vecs.append(embeds.word_vec(w))
        except KeyError:
            # Ignore a word if it's not in word2vec
            printv('Warning: %s not in vocabulary, ignoring...' % w)
            pass

    if len(vecs) == 0:
        printv('Warning: entirely OOV tweet, zeroing...')
        return np.zeros(300)

    return np.mean(vecs, axis=0)


if __name__ == '__main__':
    print('Loading word2vec...')
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)

    print('Loading training data...')
    train = data.load_train()  #[:LIMIT]
    dic = data.load_dic()

    print('Computing tweet averages...')
    X = np.zeros(shape=(len(train), 300))
    y = np.zeros(shape=(len(train),), dtype=int)
    for i, tweet in enumerate(train):
        X[i] = tweet_embedding_by_average(tweet[0], dic, embeddings)
        y[i] = tweet[1]

    print('Training the model...')
    clf = RandomForestClassifier(n_estimators=200, max_depth=10)
    clf.fit(X, y)

    print('Loading test data...')
    test = data.load_test()  #[:LIMIT]
    T = np.zeros(shape=(len(test), 300))
    ids = np.zeros(shape=(len(test),))
    for i, tweet in enumerate(test):
        T[i] = tweet_embedding_by_average(tweet[0], dic, embeddings)
        ids[i] = tweet[1]

    print('Predicting...')
    Yhat = clf.predict(T)
    data.generate_submission(Yhat, ids)
