"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""

import json
import os

import tensorflow as tf


def create_datasets(dataset, num_valid=0, max_epochs=1, batch_size=128, cache=True, **kwargs):
  """
  Takes `.tfrecord` files and creates `tf.data.Dataset` objects.

  If `dataset` is `hdfs://<path>/data/` or `hdfs://<path>/data.tfrecord`, then there should also be
  a JSON file called `hdfs://<path>/data.json` which describes the features and provides additional
  meta information. The format of the JSON file should be, for example::

      {
        "features": {
          "embedding": {"shape": [N], "dtype": "float32" },
          "author_id": { "shape": [], "dtype": "int64" },
          "author_topic": { "shape": [], "dtype": "string" },
          "item_id": { "shape": [], "dtype": "int64" },
          "item_topic": { "shape": [], "dtype": "string" },
        },
        "meta": {
          "embedding_dim": 300,
          "topics": ["tv", "politics", "sports"],
        }
      }

  If `dataset` is a path to a directory, all `.tfrecord` files in the directory will be loaded.

  Args
  ----
  dataset: A string pointing to a folder or .tfrecord file
  num_valid: If > 0, split data into training and validation sets
  max_epochs: Training data will be iterated this many times
  batch_size: How many data points to return at once
  cache: Keep dataset in memory to speed up epochs

  Returns
  -------
  One or two `tf.data.Dataset` objects and a dictionary containing meta information
  """

  # load meta information
  if dataset.lower().endswith('.tfrecord'):
    meta_info_file = dataset[:-9] + '.json'
  else:
    meta_info_file = dataset.rstrip('/') + '.json'

  with tf.gfile.GFile(meta_info_file, 'r') as handle:
    meta_info = json.load(handle)
    meta_info, features = meta_info['meta'], meta_info['features']

  # extract description of features present in the dataset
  for name, kwargs in features.items():
    features[name] = tf.FixedLenFeature(**kwargs)

  # turn serialized example into tensors
  def _parse_function(serialized):
    return tf.parse_single_example(serialized=serialized, features=features)

  if dataset.endswith('.tfrecord'):
    files = [dataset]
  else:
    files = tf.gfile.Glob(os.path.join(dataset, '*.tfrecord'))

  dataset = tf.data.TFRecordDataset(files)
  dataset = dataset.map(_parse_function, num_parallel_calls=8)

  if num_valid > 0:
    # split dataset into training and validation sets
    dataset_valid = dataset.take(num_valid)
    dataset_train = dataset.skip(num_valid)

    if cache:
      dataset_valid = dataset_valid.cache()
      dataset_train = dataset_train.cache()

    # take into account hyperparameters
    dataset_train = dataset_train.shuffle(10000).repeat(max_epochs).batch(batch_size)
    dataset_valid = dataset_valid.batch(batch_size)

    return dataset_train, dataset_valid, meta_info

  else:
    if cache:
      dataset = dataset.cache()
    dataset = dataset.shuffle(1000).repeat(max_epochs).batch(batch_size)
    return dataset, meta_info
