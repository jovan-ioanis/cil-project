The data should be placed in the ../data folder (relative to this file). We
really only need the train\_pos\_full.txt, train\_neg\_full.txt and
test\_data.txt files. train\_neg.txt and train\_pos.txt are subsets of the full
data, containing only 100,000 tweets each, so we can probably ignore them for
now.

The format is:

  * For the train files: one tweet per line. No labels (the file indicates which
    label a given tweet is.
  * For the test file: A line contains a numerical ID, a comma and then a tweet
    on each line.

For the baseline I also downloaded the pretrained Word2vec embeddings from
Google that are available here:
  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM

Place the file in the ../data folder. You will need the python package "gensim"
to load these embeddings.

`data.py` contains code related to loading and using the data. Only once, you
need to run the function rewrite in that file; it generates word IDs from the
tweets and pickles them so that loading the data is faster than reading the
whole text again. So do, in a python session:

  ``
  >>> import data
  >>> data.rewrite()
  ``

You can observe how this is used in the `baseline.py` file. That file implements
the baseline, which is to simply average the word embeddings of a tweet to get
an embedding for the tweet itself.
