"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle


def load_data(data_dir, filename):
    with tf.gfile.Open(os.path.join(data_dir, filename), 'rb') as f:
        data_dict = pickle.load(f)
        x = np.array(data_dict['x'])
        y = np.array(data_dict['y'])
        topic_names = data_dict['topic_names']
    return x, y, topic_names


def load_glove_words(data_dir):
    words = []
    with tf.gfile.Open(os.path.join(data_dir, 'glove.6B.300d.txt')) as f:
        for line in f:
            values = line.split()
            words.append(values[0])
    return words


def load_glove_embeddings(data_dir):
    word2embedding = {}
    print('Loading Glove embeddings from glove.6B.300d.txt')
    with tf.gfile.Open(os.path.join(data_dir, 'glove.6B.300d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word2embedding[word] = coefs
        print('Found %s word vectors.' % len(word2embedding))
    return word2embedding


def create_embedding_matrix(word2embedding, word2idx):
    embedding_dim = len(next(iter(word2embedding.values())))
    vocab_dim = len(word2idx)
    print('embedding_dim', embedding_dim)
    print('vocab_dim', vocab_dim)

    embedding_matrix = np.zeros((vocab_dim, embedding_dim))
    for word, i in word2idx.items():
        embedding_vector = word2embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return np.float32(embedding_matrix)


def create_1hot_embedding_matrix(word2idx):
    vocab_size = len(word2idx)
    embedding_matrix = np.eye(vocab_size)
    return np.float32(embedding_matrix)


def map_word2embedding(x, embedding_matrix):
    if isinstance(x, list):
        return [embedding_matrix[x_i] for x_i in x]

    return embedding_matrix[x]


def map_sentence2embedding(x, embedding_matrix):
    return np.mean([embedding_matrix[w] for w in x], axis=0)


def shuffle_datasets(x, y):
    shuffled_xy = shuffle(x, y, random_state=42)
    return shuffled_xy[0], shuffled_xy[1]


def make_validation_split(x, y, num_valid):
    assert num_valid < 1.  # num valid has to be a proportion

    if num_valid > 0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_valid, random_state=42)
        for t, v in sss.split(x, y):
            train_idxs, valid_idxs = list(t), list(v)
    else:
        rng = np.random.RandomState(42)
        train_idxs = list(range(len(x)))
        rng.shuffle(train_idxs)
        valid_idxs = []
    return train_idxs, valid_idxs


def remove_empty_docs_from_indexing(x, train_idxs, valid_idxs, word2idx):
    nonempty_valid_idxs = []
    nonempty_train_idxs = []

    for i in valid_idxs:
        words = [word2idx[w] for w in x[i] if w in word2idx]
        if len(words) > 0:
            nonempty_valid_idxs.append(i)

    for i in train_idxs:
        words = [word2idx[w] for w in x[i] if w in word2idx]
        if len(words) > 0:
            nonempty_train_idxs.append(i)

    return nonempty_train_idxs, nonempty_valid_idxs


def make_vocabulary(x, word2idx, vocab_size, glove_words=None):
    if word2idx is None:
        x_str = []
        for doc in x:
            if all(isinstance(s, list) for s in doc):
                for s in doc:
                    x_str.append(' '.join(s))
            else:
                x_str.append(' '.join(doc))

        print('starting count vectorizer')
        vectorizer = CountVectorizer()
        counts = np.sum(vectorizer.fit_transform(x_str).toarray(), axis=0)
        words = vectorizer.get_feature_names()
        zipped = zip(words, counts)
        zipped = sorted(zipped, key=lambda t: t[1], reverse=True)
        vocab_size = len(zipped) if vocab_size == 0 else vocab_size

        if glove_words is None:
            vocab_words = [zipped[i][0] for i in range(vocab_size)]
        else:
            words = []
            i = 0
            while len(words) < vocab_size and i < len(zipped):
                w = zipped[i][0]
                i += 1
                if w in glove_words:
                    words.append(w)
            vocab_words = words
        print('Vocabulary size:', len(vocab_words))

        idx = 0
        word2idx, idx2word = {}, {}
        for w in vocab_words:
            word2idx[w] = idx
            idx2word[idx] = w
            idx += 1
    else:
        idx2word = {}
        for (k, v) in word2idx.items():
            idx2word[v] = k
    return word2idx, idx2word
