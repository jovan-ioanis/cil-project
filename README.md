# Computational Intelligence Lab 2017
## Project: Sentiment Analysis on Tweets

This repository contains final submission for Computational Intelligence Lab 2017, ETH Zurich project about Sentiment Analysis on tweets. The authors are given below. In root folder is the baseline (Paragraph Model) and in folder `ac1d` is the improved version based on recurrent neural networks with attention model. The project is implemented using Tensorflow library.

The final submission was trained on 2.5 million tweets. The data was provided by ETH Computational Intelligence Lab staff and cannot be disclosed.

### Authors:

- [Jovan Nikolic](https://github.com/jovan-ioanis) (jovan.nikolic@gess.ethz.com)
- [Jovan Andonov](https://github.com/ac1dxtrem) (andonovj@student.ethz.ch)
- Frederic Lafrance (flafranc@student.ethz.ch)

### Project Report:

Can be found [here](https://github.com/jovan-ioanis/cil-project/blob/master/cil-project-sentiment.pdf).


### Prerequisites:

- Python 3.5.2
- Tensorflow 1.0.0
- gensim 1.0.1
- numpy 1.12.1
- tqdm 4.14.0
- nltk 3.2.2

This script expects:
1. `./data` folder containing:
   - train\_neg\_full.txt
   - train\_pos\_full.txt
   - train_neg.txt
   - train_pos.txt
   - test_data.txt
   - gnews-vecs-neg300.bin

The first five files, you have. The Google News pretrained embeddings can be found here:
`https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit`

2. `./logs` folder containing:
   - exp-1-training-2017-06-19_04-36-12
     * `exp-1-2017-06-19_04-36-12-ep-4.ckpt-78000.meta`
     * `exp-1-2017-06-19_04-36-12-ep-4.ckpt-78000.index`
     * `exp-1-2017-06-19_04-36-12-ep-4.ckpt-78000.data-00000-of-00001`
   - exp-1-training-2017-06-27_13-41-06
     * `exp-1-2017-06-27_13-41-06-ep-4.ckpt-100500.meta`
     * `exp-1-2017-06-27_13-41-06-ep-4.ckpt-100500.index`
     * `exp-1-2017-06-27_13-41-06-ep-4.ckpt-100500.data-00000-of-00001`

3. `./pickled_vars` folder containing:
   - index2word.p
   - vocab.p
   - word2index.p

4. `./word2vec` folder containing:
   - word\_embeddings\_full\_200.word2vec
   - word\_embeddings\_full\_200.word2vec.syn1neg.npy
   - word\_embeddings\_full\_200.word2vec.wv.syn0.npy

The logs, pickled_vars and word2vec folders can be found here:
`https://drive.google.com/open?id=0B2Cv2-ukPoJrTEt6SFhVQXFmSGM`

Format of the data must be one tweet per line, separate files for positives and negative tweets.

### Parameters for RNN model:

All important parameters are given in `ac1d/configuration.py` class.


### Running:

**Running the advanced model:**

In order to train seq2seq neural network, use the following command:

`python3 ac1d/main.py -n <num_cores>`

where `<num_cores>` indicates the number of cores used in Tensorflow (indicates level of parallelism). Running this script will also make predictions on test set.


The output of this run will be:

- `pickled_vars` folder with the following content:
   - `vocab.p`
   - `word2index.p`
   - `index2word.p`
- `logs` folder with saved graphs of the trained network
- `submissions` folder with submission.csv file


**Running the baseline model:**

To run the baseline model, use the following command:

`python3 baseline.py`


