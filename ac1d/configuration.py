"""
    Team OpinionatedAnalysts.
        
    Configuration file with all hyperparameters
"""


class Config:

    ###
    # FILE PATHS
    ###
    use_full_dataset = False

    DATA_ROOT = 'data'                                   # Where the data is located
    TRAIN_RAW_FULL = DATA_ROOT + '/train_%s_full.txt'    # Files with training data (positive or negative) - full
    TRAIN_RAW = DATA_ROOT + '/train_%s.txt'              # Files with 100 000 tweets
    TEST_RAW = DATA_ROOT + '/test_data.txt'              # Test data (for submission)
    PICKLED_VARS = 'pickled_vars'                        # w2i, i2w, vocab pickles
    VOCAB_PICKLED = PICKLED_VARS + '/' + 'vocab.p'
    W2I_PICKLED = PICKLED_VARS + '/' + 'word2index.p'
    I2W_PICKLED = PICKLED_VARS + '/' + 'index2word.p'

    ###
    # TOKENS
    ###
    BOS = '<bos>'
    EOS = '<eos>'
    PAD = '<pad>'
    UNK = '<unk>'
    MAX_TWEET_LENGTH = 40

    ###
    # VOCAB & WORD EMBEDDINGS
    ###
    use_word2vec = True
    VOCABULARY_SIZE = 20000
    word2vec_embedding_size = 200
    if use_full_dataset:
        word2vec_training_paths = ['data/train_pos_full.txt', 'data/train_neg_full.txt', 'data/test_data.txt']
    else:
        word2vec_training_paths = ['data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt']
    word2vec_ROOT = 'word2vec'
    if use_full_dataset:
        word2vec_filepath = word2vec_ROOT + "/" + 'word_embeddings_full_' + str(word2vec_embedding_size) + '.word2vec'
    else:
        word2vec_filepath = word2vec_ROOT + "/" + 'word_embeddings_small_' + str(word2vec_embedding_size) + '.word2vec'
    word2vec_min_frequency = 1
    word2vec_num_workers = 4

    ###
    # CELL
    ###
    cell_size = 512
    num_layers = 2
    num_epochs = 20
    batch_size = 100
    num_classes = 2
    learning_rate = 1e-4

    ###
    # CLASSIFICATION (for submission file)
    ###
    positive = 1
    negative = -1

    ###
    # ANALYSIS
    ###
    summary_frequency = 10           # validation and training accuracy and loss logs
    checkpoint_frequency = 500       # model saving 
    LOG_DIRECTORY = 'logs/'
    checkpoint_basename = 'exp-{}'
    experiment = 1
