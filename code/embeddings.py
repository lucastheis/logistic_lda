"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

This module implements embeddings so they can be reused by different models.
"""

import numpy as np
import tensorflow as tf


def identity(features, **kwargs):
  """
  Assumes embeddings are already present in the data.
  """
  return features


def one_hot(features, meta_info, **kwargs):
  """
  Assumes data contains individual words and computes one-hot encoding.
  """
  num_words = len(meta_info['words'])

  # maps words to indices
  table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(
      keys=meta_info['words'],
      values=np.arange(num_words, dtype=np.int64)), -1)

  # map words to indices, and indices to one-hot encoded embeddings
  features['embedding'] = tf.one_hot(table.lookup(features['word']), num_words)

  return features
