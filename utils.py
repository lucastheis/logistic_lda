"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import tensorflow as tf


def softmax_cross_entropy(targets, logits):
    """
    Implements a simple softmax cross entropy.

    $$-\sum_i t_{ni} \cdot (l_{ni} - \ln \sum_j \exp l_{nj})$$

    Targets can be arbitrary vectors and do not have to be one-hot encodings.

    Args:
      targets: A float tensor of shape [B, K]
      logits: A float tensor of shape [B, K]

    Returns:
      A float tensor of shape [B]
    """

    logprobs = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)
    return -tf.reduce_sum(targets * logprobs, axis=1)


def cross_entropy(targets, probs, eps=1e-12):
    logprobs = tf.log(probs + eps)
    return -tf.reduce_sum(targets * logprobs, axis=1)


def create_table(keys, values=None, name=None):
    """
    Creates a hash table which maps the given keys to integers.
    Args:
      keys: A list containing possible keys
      values: An list of corresponding values (optional)
      name: A name for the operation (optional)
    Returns:
      A `tf.contrib.lookup.HashTable` mapping keys to integers
    """

    if values is None:
        values = np.arange(len(keys), dtype=np.int64)

    return tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=values), -1, name=name)


def accuracy(preds, labels, weights=None):
    if weights is None:
        weights = np.ones_like(preds)

    # remove unlabeled data
    preds, labels, weights = remove_unlabelled_entries(preds, labels, weights)

    if len(labels) == 0:
        return 0.0

    return np.sum([w * (p == l) for w, p, l in zip(weights, preds, labels)]) / np.sum(weights)


def remove_unlabelled_entries(*lists):
    """
    Removes elements from both lists if element in one list is negative.
    """

    def _all_non_neg(values):
        return np.all(np.asarray(values) >= 0)

    lists_clean = list(zip(*filter(_all_non_neg, zip(*lists))))

    if len(lists_clean) > 1:
        return lists_clean

    return [[] for _ in lists]
