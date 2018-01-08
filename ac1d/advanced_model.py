"""
    Team OpinionatedAnalysts.

        Advanced Model:
            1. GRU cell in conf.num_layers layers
            2. projecting down to 2 classes
            3. clipping gradients larger than 10
            4. Adam optimizer
            5. loss function = sparse_softmax_cross_entropy_with_logits
            6. accuracy is number of correctly guessed sentiments vs total attempts in guessing

"""
from configuration import Config as conf
from gensim import models
import numpy as np
import tensorflow as tf

class AdvancedModel:

    def __init__(self):
        self.vocabulary_size = conf.VOCABULARY_SIZE
        self.cell_size = conf.cell_size
        self.batch_size = conf.batch_size
        self.sequence_length = conf.MAX_TWEET_LENGTH
        self.word_embedding_size = conf.word2vec_embedding_size
        self.num_layers = conf.num_layers
        self.num_classes = conf.num_classes
        self.learning_rate = conf.learning_rate
        tf.reset_default_graph()

    def init_placeholders(self):
        with tf.variable_scope('DATA'):
            # self.batch_size_tensor = tf.placeholder(tf.int32, shape=(), name='batch_size')
            self.x = tf.placeholder(tf.int32, [self.sequence_length, self.batch_size], name='x')  # [batch_size, sequence_length]
            # self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length_decoder')
            self.y = tf.placeholder(tf.int32, [self.batch_size, self.num_classes], name='y')  # [batch_size, 2]

    def init_embeddings(self):
        with tf.variable_scope("word_embedding"):
            self.W_embed = tf.get_variable(name='W_embed', shape=[self.vocabulary_size, self.word_embedding_size], initializer=tf.contrib.layers.xavier_initializer())  # [vocabulary_size, word_embedding_size]
            self.embeddings = tf.nn.embedding_lookup(self.W_embed, self.x)  # [batch_size, bucket_size, word_embedding_size]

    def load_embeddings(self, dictionary, session):
        print("Loading external embeddings from %s" % conf.word2vec_filepath)
        model = models.Word2Vec.load(conf.word2vec_filepath)
        external_embedding = np.zeros(shape=(self.vocabulary_size, self.word_embedding_size))
        matches = 0
        for idx, tok in enumerate(dictionary.keys()):
            if tok in model.wv.vocab:
                external_embedding[idx] = model[tok]
                matches += 1
            else:
                print("%s not in embedding file" % tok)
                external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=self.word_embedding_size)

        print("%d words out of %d could be loaded" % (matches, self.vocabulary_size))

        pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
        assign_op = self.W_embed.assign(pretrained_embeddings)
        session.run(assign_op, {pretrained_embeddings: external_embedding})  # here, embeddings are actually set

    def init_cell(self):
        self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.cell_size) for _ in range(self.num_layers)])

    def dynamic_rnn(self):
        with tf.variable_scope('RNN'):
            self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.embeddings, dtype=tf.float32, time_major=True)
            print(self.outputs.shape)

    def attention(self):
        with tf.variable_scope('Attention'):

            self.attention_inputs = tf.reshape(self.outputs, [self.sequence_length, -1])  # [sequence_length, batch_size * cell_size]

            self.W_attention = tf.get_variable('W_attention', [self.batch_size*self.cell_size, 1],  # [batch_size * cell_size, 1]
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b_attention = tf.get_variable('b_attention', [self.sequence_length, 1], initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable_scope().reuse_variables()

            self.attention_logits = tf.matmul(self.attention_inputs, self.W_attention) + self.b_attention  # [sequence_length, 1]
            self.attention_weights = tf.nn.softmax(self.attention_logits)  # [sequence_length, 1]
            self.attention_weights = tf.transpose(self.attention_weights)
            print('Attention weights shape: {}'.format(self.attention_weights.shape))

            print('Outputs shape: {}'.format(self.outputs.shape))
            self.outputs = tf.matmul(self.attention_weights, tf.reshape(self.outputs, [self.sequence_length, -1]))
            self.outputs = tf.reshape(self.outputs, [self.batch_size, self.cell_size])
            print('Outputs shape: {}'.format(self.outputs.shape))


    def softmax(self):
        with tf.variable_scope('Logits', reuse=None):
            self.W = tf.get_variable('W', [self.cell_size, self.num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable('b', [self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable_scope().reuse_variables()

            # self.logits = tf.matmul(tf.reduce_mean(self.outputs, 0), self.W) + self.b  # [batch_size, num_classes]
            self.logits = tf.matmul(self.outputs, self.W) + self.b  # [batch_size, num_classes]
            print("logits shape is: {}".format(self.logits.get_shape()))
            # self.prediction_probs = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.logits, 1)

    def losses(self):
        with tf.variable_scope('LOSS'):
            self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y,
                logits=self.logits,
            )

    def accuracy(self):
        with tf.variable_scope('ACCURACY'):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.prediction_accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name='accuracy')

    def optimization_step(self):
        with tf.variable_scope('TRAIN'):
            self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=10)
            self.train_step = optimizer.apply_gradients(zip(grads, tvars))

    def summary(self):
        with tf.variable_scope('SUMMARY'):
            tf.summary.scalar("mean loss", self.loss)
            tf.summary.scalar("accuracy", self.prediction_accuracy)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self.init_placeholders()
        self.init_embeddings()
        self.init_cell()
        self.dynamic_rnn()
        self.attention()
        self.softmax()
        self.losses()
        self.accuracy()
        self.optimization_step()
        self.summary()

