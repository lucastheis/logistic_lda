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

  Targets can be arbitrary vectors and do not have to be one-hot encodings or normalized,
  unlike in some other implementations of cross-entropy.

  Args:
    targets: A float tensor of shape [B, K]
    logits: A float tensor of shape [B, K]

  Returns:
    A float tensor of shape [B]
  """

  logprobs = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)
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
