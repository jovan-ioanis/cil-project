import data
import gensim
import numpy as np

WORD2VEC = '../data/gnews-vecs-neg300.bin'
#LIMIT = None # NO LIMITS
LIMIT = 10000
VERBOSE = True

cfg = {
    'vocab_size': 20000, # How many words in our vocabulary
    'num_pars': LIMIT, # How many different paragraphs (i.e. tweets) do we train

    'pv_context': 2, # How many words of context to use
    'pv_embedding_size': 300, # Size of word/paragraph embeddings
    'pv_num_neg_samples': 20, # How many negative samples to use in training
}

def make_training_contexts(tweets, dic)
    '''
    Given tweets, create training contexts out of them. Concretely, if a tweet
    is of the form:
        w1, w2, w3, ... wn
    We want to generate:
        <s_1> ...      <s_c> | w1
        <s_1> ... <s_c-1> w1 | w2
        ...
        w(t-c) ... w(t-1) | wt
        ...
        w(n-c) ... w(n-1) | wn

    In other words, we generate pairs of (context, word) for every word in the
    tweet, where `context` is a list of the c preceding words in the tweet,
    including start symbols if there are less than c preceding words.
    '''

    total_count = sum([len(x) for x in tweets])
    X = np.empty((total_count, cfg['pv_context']), dtype=int)
    Y = np.empty(total_count, dtype=int)

    total_i = 0
    for t in tweets:
        # Add the minimal amount of pad (always at least 1 real word in context)
        t = ([dic.word_to_id['<pad>']] * (cfg['pv_context'] - 1)) + t
        for i in range(cfg['pv_context'], len(t)):
            X[total_i] = np.array(t[i - cfg['pv_context']:i])
            Y[total_i] = t[i]

    return (X, Y)

def make_embedding_matrix(dic, embeds):
    '''
    Construct the embedding matrix out of the given dictionary and the given
    pretrained embeddings.

    If a word is in the dictionary but not in the embeddings, a random vector is
    used as that row of the matrix.

    Return a matrix of size vocab_size x embedding_size.
    '''

    mat = np.empty((len(dic.id_to_word), cfg['pv_embedding_size']))
    for i, w in enumerate(dic.id_to_word):
        if w in embeds:
            mat[i] = embeds.word_vec(w)
        else:
            mat[i] = np.random.uniform(-1.0, 1.0, cfg['pv_embedding_size'])

    return mat



if __name__ == '__main__':
    # Decide which mode we are in
    #   train_pv -> trains and saves the paragraph vector model
    #   infer_pv -> holding the model fixed, infers new paragraph vectors
    #   classify -> Train a model on the paragraph vectors and then classify.
    #
    #   Still TODO: intelligently split our data.
    #   The experimental protocol in the paper is as follows:
    #
    #   Train the paragraph vectors, and use those to train a classifier
    #   When testing:
    #      - hold the previously learned weights fixed, and infer paragraph vecs
    #      - feed these inferred vecs to the trained classifier.
    #
    pass
