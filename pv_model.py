import tensorflow as tf

class ParagraphVectorModel(object):
    '''
    Implementation of the paragraph vector, based on [0].

    This model is at its core a language model, where we predict the next word
    of a sentence given some words of context. These words are converted into
    embeddings by a shared matrix. We also provide as input a paragraph ID,
    which is itself converted into an embedding. Those embeddings are then
    averaged to produce a context embedding, which is then decoded into the
    final probabilities for the word.

    Training is performed using SGD.

    To obtain a new paragraph vector, we perform the same procedure on the words
    of the paragraph while keeping their embeddings fixed. This gradient
    descent yields a paragraph vector.

    This model corresponds to the "distributed memory" model described in [0],
    which achieves better results than the distributed bag-of-words model.

    The code itself takes inspiration from [1].

    [0] Le and Mikolov (2014). Distributed representations of sentences and documents.
    [1] https://www.tensorflow.org/tutorials/word2vec
    '''


    def __init__(self, cfg, embeddings=None):
        self.cfg = cfg
        self.embeddings = embeddings

        self.model_session = tf.Session(config=tf.ConfigProto())

    def save_model(self, path):
        print("Model saved in file: %s" % save_path)
        saver = tf.train.Saver()
        save_path = saver.save(self.model_session, path)

    def _build_train(self):
        print('PV: building forward training phase...')

        xavier_init = tf.contrib.layers.xavier_initializer()

        # Inputs: context word IDs, paragraph IDs, target word IDs
        # 'None' represents the batch size here
        self.words = tf.placeholder(tf.int32, [None, self.cfg['pv_context']])
        self.paragraphs = tf.placeholder(tf.int32, [1, None])
        self.targets = tf.placeholder(tf.int32, [None])

        # Word in-embedding matrix
        W_in_sz = [self.cfg['vocab_size'], self.cfg['pv_embedding_size']]
        if self.embeddings is not None:
            W_in = tf.Variable(self.embeddings, name='W_in', expected_shape=W_in_sz)
        else:
            W_in = tf.get_variable('W_in', W_in_sz, tf.float32, xavier_init)

        # Word in-embedding bias
        b_in = tf.get_variable('b_in', [self.cfg['pv_embedding_size']],
                                tf.float32, xavier_init)

        # Paragraph in-embedding matrix and bias
        P_in_sz = [self.cfg['num_pars'], self.cfg['pv_embedding_size']]
        P_in = tf.get_variable('P_in', P_in_sz, tf.float32, xavier_init)
        bp_in = tf.get_variable('bp_in', [self.cfg['pv_embedding_size']],
                                tf.float32, xavier_init)

        # Combined out-embedding matrix and bias
        W_out_sz = [self.cfg['vocab_size'], self.cfg['pv_embedding_size']]
        W_out = tf.get_variable('W_out', W_out_sz, tf.float32, xavier_init)
        b_out = tf.get_variable('b_out', [self.cfg['vocab_size']],
                                tf.float32, xavier_init)


        # batch_size x context_size x embedding_size
        w_embeds = tf.nn.embedding_lookup(W_in, self.words) + b_in

        # batch_size x 1 x embedding_size
        p_embeds = tf.nn.embedding_lookup(P_in, self.paragraphs) + bp_in

        # batch_size x (context_size + 1) x embedding_size
        wp_embeds = tf.concat([w_embeds, p_embeds], axis=1)

        #batch_size x embedding_size
        wp_embeds_avg = tf.reduce_mean(wp_embeds, axis=1)

        # Use noise-contrastive estimation, which is similar to negative
        # sampling for softmax (this is from the tutorial)

        loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=W_out,
                                biases=b_out,
                                labels=self.targets,
                                inputs=wp_embeds_avg,
                                num_sampled=self.cfg['pv_num_neg_samples'],
                                num_classes=self.cfg['vocab_size']))

        self.train_op = tf.train.AdamOptimizer().minimize(loss)

    def train(self):
        '''
        Jointly train paragraph and word embeddings.
        '''
        pass

    def infer(self):
        '''
        Use the previously trained word embeddings to compute the paragraph
        vector for a new sequence of words.
        '''
        pass

