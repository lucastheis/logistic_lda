"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

TF datasets for Newsgroups 20 with batches containing full documents.

"""

from collections import defaultdict

import numpy as np
import tensorflow as tf

import data_utils


def create_tf_dataset(data_dir, filename, num_valid=0, max_epochs=None,
                      vocab_size=None, word2idx=None, embedding=None, batch_size=1):
    print('Using batches of full documents')

    glove_words = data_utils.load_glove_words(data_dir) if embedding == 'glove' else None

    features, meta_info, word2idx = create_data_record(data_dir, filename, num_valid, word2idx,
                                                       vocab_size, glove_words)

    def generator_train():
        for i in meta_info['train_features_idxs']:
            features_i = {
                'embedding': features['embedding'][i],
                'author_id': features['author_id'][i],
                'author_topic': features['author_topic'][i],
                'item_id': features['item_id'][i],
                'item_topic': features['item_topic'][i]
            }
            yield features_i

    dataset_train = tf.data.Dataset.from_generator(
        generator_train, output_types={
            'embedding': tf.int64,
            'author_id': tf.int64,
            'author_topic': tf.string,
            'item_id': tf.int64,
            'item_topic': tf.string}, output_shapes={'embedding': tf.TensorShape([None]),
                                                     'author_id': tf.TensorShape([None]),
                                                     'author_topic': tf.TensorShape([None]),
                                                     'item_id': tf.TensorShape([None]),
                                                     'item_topic': tf.TensorShape([None])}
    )

    dataset_train = dataset_train.shuffle(1000).repeat(max_epochs)

    dataset_valid = None
    if num_valid > 0:
        def generator_valid():
            for i in meta_info['valid_features_idxs']:
                features_i = {
                    'embedding': features['embedding'][i],
                    'author_id': features['author_id'][i],
                    'author_topic': features['author_topic'][i],
                    'item_id': features['item_id'][i],
                    'item_topic': features['item_topic'][i]
                }
                yield features_i

        dataset_valid = tf.data.Dataset.from_generator(
            generator_valid, output_types={
                'embedding': tf.int64,
                'author_id': tf.int64,
                'author_topic': tf.string,
                'item_id': tf.int64,
                'item_topic': tf.string}, output_shapes={'embedding': tf.TensorShape([None]),
                                                         'author_id': tf.TensorShape([None]),
                                                         'author_topic': tf.TensorShape([None]),
                                                         'item_id': tf.TensorShape([None]),
                                                         'item_topic': tf.TensorShape([None])}
        )

    return dataset_train, dataset_valid, meta_info, word2idx


def create_data_record(data_dir, filename, num_valid, word2idx, vocab_size, glove_words):
    print('loading data')
    x, y, topic_names = data_utils.load_data(data_dir, filename)
    x, y = data_utils.shuffle_datasets(x, y)
    topic_name2number = dict(zip(topic_names, range(len(topic_names))))
    topic_number2name = dict(zip(range(len(topic_names)), topic_names))

    # make a vocabulary of size vocab_size or use the one that is given
    word2idx, idx2word = data_utils.make_vocabulary(x, word2idx, vocab_size, glove_words)
    vocab_size = len(word2idx)

    # split documents into train and valid
    train_idxs, valid_idxs = data_utils.make_validation_split(x, y, num_valid)

    # filter out empty docs
    train_idxs, valid_idxs = data_utils.remove_empty_docs_from_indexing(x, train_idxs, valid_idxs, word2idx)

    features, meta_info = make_data_dict(x, y, word2idx, train_idxs, valid_idxs, topic_number2name,
                                         topic_names)
    # add data path, so we can load glove embedding if needed
    meta_info['data_dir'] = data_dir

    print('---- Number of docs per topic ----')
    for t in topic_names:
        print('%s: %s' % (t, len(np.where(y == topic_name2number[t])[0])))
    print('------------------------------------')

    print('n docs', len(train_idxs + valid_idxs))
    print('n train docs', len(train_idxs))
    print('n valid docs', len(valid_idxs))
    print('n items:', len(features['item_id']))
    print('vocab size', vocab_size)

    print('max items per doc', np.max(list(meta_info['author2nitems'].values())))
    print('min items per doc', np.min(list(meta_info['author2nitems'].values())))
    print('avg items per doc', np.mean(list(meta_info['author2nitems'].values())))
    print('topics', meta_info['topics'])

    return features, meta_info, word2idx


def make_data_dict(x, y, word2idx, train_idxs, valid_idxs, topic_number2name, selected_topics):
    embeddings = []
    author_id, author_topic = [], []
    item_id, item_topic = [], []

    item_ids_counter = 0
    author2nitems = defaultdict(lambda: 0)
    for i in valid_idxs + train_idxs:
        words = [word2idx[w] for w in x[i] if w in word2idx]
        embeddings.append(words)
        author2nitems[i] += len(words)
        author_id.append([i] * len(words))
        author_topic.append([topic_number2name[y[i]]] * len(words))
        item_id.append(list(range(item_ids_counter, item_ids_counter + len(words))))
        item_ids_counter += len(words)
        item_topic.append([''] * len(words))

    n_valid_items = len(valid_idxs)

    features = {
        'embedding': embeddings,
        'author_id': author_id,
        'author_topic': author_topic,
        'item_id': item_id,
        'item_topic': item_topic
    }

    meta_info = {'topics': selected_topics,
                 'author_ids': np.array(valid_idxs + train_idxs),
                 'author2nitems': author2nitems,
                 'n_valid_items': n_valid_items,
                 'valid_features_idxs': range(0, n_valid_items),
                 'train_features_idxs': range(n_valid_items, n_valid_items + len(train_idxs)),
                 'word2idx': word2idx}
    return features, meta_info


def create_tf_vocab_dataset(word2idx, data_dir, batch_size=16):
    """
    Dataset of single words from the vocabulary. Used for testing the unsupervised models.
    """
    embeddings = []
    author_id, author_topic = [], []
    item_id, item_topic = [], []
    author2nitems = {}

    for k, v in word2idx.items():
        embeddings.append([v])
        author2nitems[v] = 1
        author_id.append([v])
        author_topic.append([''])
        item_id.append([v])
        item_topic.append([''])

    features = {
        'embedding': embeddings,
        'author_id': author_id,
        'author_topic': author_topic,
        'item_id': item_id,
        'item_topic': item_topic
    }

    meta_info = {'topics': [],
                 'author_ids': list(author2nitems.keys()),
                 'author2nitems': author2nitems,
                 'n_valid_items': 0,
                 'valid_features_idxs': range(0),
                 'train_features_idxs': range(len(author_id)),
                 'word2idx': word2idx,
                 'data_dir': data_dir}

    def generator():
        for i in range(len(word2idx)):
            features_i = {
                'embedding': features['embedding'][i],
                'author_id': features['author_id'][i],
                'author_topic': features['author_topic'][i],
                'item_id': features['item_id'][i],
                'item_topic': features['item_topic'][i]
            }
            yield features_i

    dataset = tf.data.Dataset.from_generator(
        generator, output_types={
            'embedding': tf.int64,
            'author_id': tf.int64,
            'author_topic': tf.string,
            'item_id': tf.int64,
            'item_topic': tf.string}, output_shapes={'embedding': tf.TensorShape([None]),
                                                     'author_id': tf.TensorShape([None]),
                                                     'author_topic': tf.TensorShape([None]),
                                                     'item_id': tf.TensorShape([None]),
                                                     'item_topic': tf.TensorShape([None])})
    return dataset, meta_info, word2idx


class ConcatBatchIterator:
    """
    This iterator concatenates batches of single documents into batches of multiple docs.
    """

    def __init__(self, dataset, batch_size):
        self.single_batch_iterator = dataset.make_one_shot_iterator()
        self.batch_size = batch_size
        print('A single batch contains %s full documents' % self.batch_size)

    def get_next(self):
        features = self.single_batch_iterator.get_next()
        for i in range(self.batch_size - 1):
            next_batch = self.single_batch_iterator.get_next()
            features['embedding'] = tf.concat([features['embedding'], next_batch['embedding']], axis=0)
            features['author_id'] = tf.concat([features['author_id'], next_batch['author_id']], axis=0)
            features['author_topic'] = tf.concat([features['author_topic'], next_batch['author_topic']], axis=0)
            features['item_id'] = tf.concat([features['item_id'], next_batch['item_id']], axis=0)
            features['item_topic'] = tf.concat([features['item_topic'], next_batch['item_topic']], axis=0)
        return features
