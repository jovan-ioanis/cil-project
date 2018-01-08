"""
    Team OpinionatedAnalysts.

    Main interface for training and testing the model.
"""


import datetime
import os, sys, getopt
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math
import preprocessing, advanced_model, word_embeddings
from configuration import Config as conf


class Analyzer:
    """
        Class encapsulating all Tensorboard analysis:
        1. validation error:    every validation_frequency steps, whole batch
                                is used to just predict sentiment, not to update weights
        2. training error:      every training_frequency step, whole batch is used for both
                                training and predicting the sentiment
        3. checkpoint:          every checkpoint_frequency steps a graph is saved on disk
    """
    def __init__(self):
        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        self.summary_frequency = conf.summary_frequency
        self.checkpoint_frequency = conf.checkpoint_frequency
        self.checkpoint_name = conf.checkpoint_basename.format(1)
        self.timestamp = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        self.train_logfolderPath = \
            os.path.join(conf.LOG_DIRECTORY,
                         "{}-training-{}".format(conf.checkpoint_basename.format(conf.experiment), self.timestamp))
        self.validation_logfolderPath = \
            os.path.join(conf.LOG_DIRECTORY,
                         "{}-validation-{}".format(conf.checkpoint_basename.format(conf.experiment), self.timestamp))
        self.train_writer = \
            tf.summary.FileWriter(self.train_logfolderPath, graph=tf.get_default_graph())
        self.validation_writer = \
            tf.summary.FileWriter(self.validation_logfolderPath, graph=tf.get_default_graph())

    def validation_error(self, global_step, session, model, feed_dictionary):
            summary_validation = session.run(model.summary_op, feed_dictionary)
            self.validation_writer.add_summary(summary_validation, global_step)

    def train_error(self, global_step, session, model, feed_dictionary):
            summary_train = session.run(model.summary_op, feed_dictionary)
            self.train_writer.add_summary(summary_train, global_step)

    def save_checkpoint(self, global_step, epoch, session, force=False):
        if not force:
            name = os.path.join(self.train_logfolderPath,
                                "{}-{}-ep-{}.ckpt".format(
                                    conf.checkpoint_basename.format(conf.experiment),
                                    self.timestamp,
                                    epoch))
            self.saver.save(session, name, global_step=global_step)
        else:
            name = os.path.join(self.train_logfolderPath,
                                "{}-{}-ep-{}-final.ckpt".format(
                                    conf.checkpoint_basename.format(conf.experiment),
                                    self.timestamp,
                                    epoch))
            self.saver.save(session, name)


def train_graph(model, tweets_w_labels, dictionary, test_tweets, num_cores, load_saved=False, model_path=None):
    """
        Training Advanced Model:
            1. First 10k tweets are kept for validation, the rest is used for training
            2.
    :param model:                   built graph
    :param tweets_w_labels:         training+validation tweets with matching labels
    :param dictionary:              Dic object
    :param test_tweets:             tweets from test file
    """
    validation_tweets_w_labels = tweets_w_labels[:10000]

    tweets_w_labels = tweets_w_labels[10000:]

    if num_cores != -1:
        configProto = tf.ConfigProto(inter_op_parallelism_threads=num_cores,
                                     intra_op_parallelism_threads=num_cores)
    else:
        configProto = tf.ConfigProto()

    with tf.Session(config=configProto) as sess:
        sess.run(tf.global_variables_initializer())
        analyzer = Analyzer()

        if load_saved and model_path:
            analyzer.saver.restore(sess, model_path)
            parts = model_path.split('-')
            global_step = int(parts[len(parts)-1])
        else:
            global_step = 1

        if conf.use_word2vec:
            model.load_embeddings(dictionary.word2index, sess)

        feed_dictionary_train = {}

        for epoch in range(conf.num_epochs):
            print("\t\t\tEPOCH: {}".format(epoch))

            random.shuffle(validation_tweets_w_labels)
            validation_x = []
            validation_y = []
            for y, x in validation_tweets_w_labels:
                validation_x.append(x)
                validation_y.append(y)
            validation_x = list(map(list, zip(*validation_x)))
            validation_x = np.array(validation_x)

            print('Validation shape: {} {}', len(validation_x), len(validation_x[0]))

            for input_x, target_y in tqdm(batching(tweets_w_labels), total=math.ceil(len(tweets_w_labels) / conf.batch_size)):

                global_step += 1

                input_x = list(map(list, zip(*input_x)))  # Transpose it.
                # target_y = list(map(list, zip(*target_y)))  # Transpose it.

                feed_dictionary_train[model.x] = input_x
                feed_dictionary_train[model.y] = target_y

                if global_step % analyzer.summary_frequency == 0:
                    validation_batch = global_step % 100
                    feed_dictionary_validation = {model.x: validation_x[:, validation_batch * conf.batch_size: (validation_batch + 1) * conf.batch_size],
                                                  model.y: validation_y[validation_batch * conf.batch_size: (validation_batch + 1) * conf.batch_size]}
                    analyzer.validation_error(global_step, sess, model, feed_dictionary_validation)
                    analyzer.train_error(global_step, sess, model, feed_dictionary_train)

                if global_step % analyzer.checkpoint_frequency == 0:
                    analyzer.save_checkpoint(global_step, epoch, sess)

                sess.run([model.train_step], feed_dictionary_train)

            analyzer.save_checkpoint(global_step, epoch, sess)
            get_predictions(model, test_tweets, epoch, sess=sess, load_saved=False)

        analyzer.save_checkpoint(global_step, conf.num_epochs, sess, True)


def get_predictions(model, tweets, epoch, sess=None, load_saved=False, model_path=None):

    if not sess:
        print('Initializing a session')
        sess = tf.Session()

    print('Generating predictions from Epoch:', epoch)

    if load_saved and model_path:
        analyzer = Analyzer()
        analyzer.saver.restore(sess, model_path)

    predictions = []

    for i in range(0, 100):
        batch_predictions = sess.run(model.predictions, feed_dict={model.x: tweets[:, i*conf.batch_size : (i+1)*conf.batch_size]})
        predictions.extend(batch_predictions)

    if not os.path.exists('submissions'):
        os.makedirs('submissions')

    f = open('submissions/submission-' + str(epoch) + '.csv', 'w')
    f.write('ID,Prediction\n')

    for i, prediction in enumerate(predictions):
        if prediction == 0:
            f.write(str(i+1) + ',' + str(conf.negative) + '\n')
        else:
            f.write(str(i+1) + ',' + str(conf.positive) + '\n')
    f.close()


def batching(tweets_w_labels):
    """
        1. Shuffling data
        2. Trimming data so that every batch has the same size.
        2. yields tuple (input_tweets_batch, target_batch)
    """
    random.shuffle(tweets_w_labels)

    data_length = len(tweets_w_labels)
    print("Number of tweets for training: {}".format(data_length))
    num_batches = data_length // conf.batch_size
    print("Number of batches: {}".format(num_batches))
    r = data_length % conf.batch_size
    print("Rest is: {}".format(r))

    chosen_tweets = []
    chosen_labels = []

    for tweet_tuple in tweets_w_labels:
        chosen_labels.append(tweet_tuple[0])
        chosen_tweets.append(tweet_tuple[1])

    chosen_tweets = np.array(chosen_tweets)
    chosen_labels = np.array(chosen_labels)

    print("Number of chosen tweets is: {}".format(len(chosen_tweets)))
    for i in range(num_batches):
        current_batch_x = chosen_tweets[(i*conf.batch_size):((i+1)*conf.batch_size), :]
        current_batch_y = chosen_labels[(i*conf.batch_size):((i+1)*conf.batch_size), :]
        yield (current_batch_x, current_batch_y)


def mainFunc(argv):
    def print_usage():
        print('python3 ac1d/main.py -n <num_cores>')
        print('num_cores = Number of cores requested from the cluster. Set -1 to leave unset.')
    num_cores = -1
    try:
        opts, args = getopt.getopt(argv, "n:", ['num_cores='])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt in ('-n', '--num_cores'):
            num_cores = int(arg)
    print(">> NUMBER OF CORES USED: {} ".format(num_cores))
    tweets_w_labels, dictionary = preprocessing.load_dataset(full=conf.use_full_dataset)
    test_tweets = np.transpose(np.array(preprocessing.load_test_data(dictionary)))

    if conf.use_word2vec:
        word_embeddings.train_embeddings()

    model = advanced_model.AdvancedModel()
    model.build_graph()

    get_predictions(model, test_tweets, 0, load_saved=True, model_path='logs/exp-1-training-2017-06-26_21-47-32/exp-1-2017-06-26_21-47-32-ep-0.ckpt-1500')
    # train_graph(model, tweets_w_labels, dictionary, test_tweets, num_cores)

if __name__ == "__main__":
    mainFunc(sys.argv[1:])
