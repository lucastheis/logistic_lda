"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import os
import pickle
from collections import namedtuple

import numpy as np
import tensorflow as tf

from scripts import nltk_tokenizer


def _read_text_file(path):
    with open(path, 'rt', encoding='utf-8') as file:
        # Read a list of strings.
        lines = file.readlines()
        # Concatenate to a single string.
        text = " ".join(lines)
    return text


def save_data(x, y, topic_names, data_dir, filename):
    with open(os.path.join(data_dir, filename), 'wb') as f:
        pickle.dump({'x': x, 'y': y, 'topic_names': topic_names}, f)


def load_data(data_dir, filename):
    with tf.gfile.Open(os.path.join(data_dir, filename), 'rb') as f:
        data_dict = pickle.load(f)
        x = data_dict['x']
        y = data_dict['y']
        topic_names = data_dict['topic_names']
    return x, y, topic_names


def read_news20_data(data_dir):
    """
    Writes Newsgroups 20 from ana.cachopo.org/datasets-for-single-label-text-categorization
    into a suitable format.
    This version was also used in arxiv.org/abs/1511.01432 and arxiv.org/abs/1602.02373
    """

    with open(os.path.join(data_dir, '20ng-train-all-terms.txt')) as f:
        x_train, y_train = [], []
        for line in f:
            l = line.split('\t', 1)
            y_train.append(l[0])
            x_train.append(l[1])

    target_names = list(np.unique(y_train))
    y_train = [target_names.index(y_i) for y_i in y_train]

    with open(os.path.join(data_dir, '20ng-test-all-terms.txt')) as f:
        x_test, y_test = [], []
        for line in f:
            l = line.split('\t', 1)
            y_test.append(l[0])
            x_test.append(l[1])
    y_test = [target_names.index(y_i) for y_i in y_test]

    return namedtuple('Data', ['data', 'target', 'target_names'])(x_train, y_train, target_names), \
           namedtuple('Data', ['data', 'target', 'target_names'])(x_test, y_test, target_names)


def run_nltk_word_tokenizer(data, remove_stop_words=True):
    preprocessor = nltk_tokenizer.NLTKPreprocessor()

    x, y = [], []
    n_empty_docs = 0
    for i, doc in enumerate(data.data):
        tokenized_text = preprocessor.tokenize(doc, remove_stop_words)
        if i % 1000 == 0:
            print(i)
        if len(tokenized_text) == 0:
            n_empty_docs += 1
        else:
            x.append(tokenized_text)
            y.append(data.target[i])

    print('n total docs', len(data.data))
    print('n empty docs', n_empty_docs)

    return x, y


def preprocess_ng20(data_dir):
    train_data, test_data = read_news20_data(data_dir)
    topic_names = train_data.target_names

    x, y = run_nltk_word_tokenizer(train_data)
    save_data(x, y, topic_names, data_dir, 'tokenized_ng20_train.pkl')

    x, y = run_nltk_word_tokenizer(test_data)
    save_data(x, y, topic_names, data_dir, 'tokenized_ng20_test.pkl')


def make_reference_corpus_for_topic_coherence(data_dir):
    """
    Writes Newsgroups 20 into a format suitable for running topics coherence evaluation code
    from github.com/jhlau/topic_interpretability
    """
    x_train, _, _ = load_data(data_dir, 'tokenized_ng20_train.pkl')
    x_test, _, _ = load_data(data_dir, 'tokenized_ng20_test.pkl')

    docs_train = [' '.join(doc) for doc in x_train]
    docs_test = [' '.join(doc) for doc in x_test]
    all_docs = docs_train + docs_test
    with open('reference_ng20.txt', 'a') as f:
        for doc in all_docs:
            f.write(doc + '\n')


if __name__ == '__main__':
    print('Processing newsgroups 20')
    preprocess_ng20('data_debug')

    print('Making reference corpus for topics coherence')
    make_reference_corpus_for_topic_coherence('data_debug')
