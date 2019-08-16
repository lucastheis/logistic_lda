#!/usr/bin/env python3

"""
Evaluates accuracy with which a model predicts topics of items and authors.
"""

from __future__ import print_function

import csv
import inspect
import json
import os
import pprint
import sys
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial

from logistic_lda.data import create_datasets
from logistic_lda import embeddings, models

import numpy as np
import tensorflow as tf


def main(args):
  # prepare training and validation data
  dataset_test, meta_info = create_datasets(max_epochs=args.num_iter, **vars(args))

  def get_dataset_iterator():
    return dataset_test.make_one_shot_iterator().get_next()

  embedding = partial(getattr(embeddings, args.embedding), meta_info=meta_info, args=args, mode='valid')

  # infer topics and author's topic distributions (initialization)
  classifier = tf.estimator.Estimator(
      model_fn=getattr(models, args.model),
      params={
        'hidden_units': args.hidden_units,
        'learning_rate': 0.0,
        'decay_rate': 0.0,
        'decay_steps': 0,
        'alpha': args.topic_bias_regularization,
        'model_regularization': 0.0,
        'author_topic_weight': 0.0,  # this avoids local optima of MF-VI
        'author_topic_iterations': 1,
        'items_per_author': args.items_per_author,
        'meta_info': meta_info,
        'embedding': embedding,
        'use_author_topics': False,
      },
      warm_start_from=args.model_dir,
    )
  classifier.train(get_dataset_iterator)

  # infer topics and author's topic distributions
  classifier = tf.estimator.Estimator(
      model_fn=getattr(models, args.model),
      params={
        'hidden_units': args.hidden_units,
        'learning_rate': 0.0,
        'decay_rate': 0.0,
        'decay_steps': 0,
        'alpha': args.topic_bias_regularization,
        'model_regularization': 0.0,
        'author_topic_weight': args.author_topic_weight,
        'author_topic_iterations': args.author_topic_iterations,
        'items_per_author': args.items_per_author,
        'meta_info': meta_info,
        'embedding': embedding,
        'use_author_topics': False,
      },
      model_dir=classifier.model_dir,
    )
  classifier.train(get_dataset_iterator)

  author_predictions = []
  author_topics = []
  author_ids = []
  item_predictions = []
  item_probabilities = []
  item_topics = []
  item_ids = []
  tweets_per_author = defaultdict(int)

  # get labels and predictions
  for prediction in classifier.predict(get_dataset_iterator):
    author_predictions.append(prediction['author_prediction'])
    author_topics.append(prediction['author_topic'])
    author_ids.append(prediction['author_id'])
    item_predictions.append(prediction['item_prediction'])
    item_probabilities.append(prediction['item_probability'])
    item_topics.append(prediction['item_topic'])
    item_ids.append(prediction['item_id'])
    tweets_per_author[prediction['author_id']] += 1

  # used to count each author only once when calculating accuracy for author prediction
  weights = [1.0 / tweets_per_author[author_id] for author_id in author_ids]

  results = {}

  # evaluate accuracy
  results['accuracy_author'] = accuracy(author_predictions, author_topics)
  results['accuracy_author_unweighted'] = accuracy(author_predictions, author_topics, weights)
  results['accuracy_item'] = accuracy(item_predictions, item_topics)

  pprint.pprint(results)

  # save results
  if args.output_results:
    with tf.gfile.GFile(args.output_results, 'w') as handle:
      json.dump(results, handle, indent=4)

  if args.output_predictions:
    # save predictions
    with tf.gfile.GFile(args.output_predictions, 'w') as handle:
      writer = csv.writer(handle)
      writer.writerow([
        'author_id',
        'author_topic',
        'author_prediction',
        'item_id',
        'item_topic',
        'item_prediction',
        'item_probability'])

      for i in range(len(author_ids)):
        writer.writerow([
          author_ids[i],
          author_topics[i],
          author_predictions[i],
          item_ids[i],
          item_topics[i],
          item_predictions[i],
          item_probabilities[i]])

  return 0


def accuracy(preds, labels, weights=None):
  if weights is None:
    weights = np.ones_like(preds)

  # remove unlabeled data
  preds, labels, weights = remove_unlabelled_entries(preds, labels, weights)

  if len(labels) == 0:
    return 0.0

  return np.sum([w * (p == l) for w, p, l in zip(weights, preds, labels)]) / float(np.sum(weights))


def remove_unlabelled_entries(*lists):
    """
    Removes elements from all lists if element in one list is negative.
    """

    def _all_non_neg(values):
        return np.all(np.asarray(values) >= 0)

    lists_clean = list(zip(*filter(_all_non_neg, zip(*lists))))

    if len(lists_clean) > 1:
        return lists_clean

    return [[] for _ in lists]


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', type=str,
      help='Path to a TFRecord dataset')
  parser.add_argument('--batch_size', type=int, default=None,
      help='Number of items per training batch')
  parser.add_argument('--num_iter', type=int, default=1,
      help='Number of passes through the dataset to compute beliefs with variational inference')
  parser.add_argument('--topic_bias_regularization', type=float, default=None,
      help='Parameter of Dirichlet prior on topic proportions')
  parser.add_argument('--items_per_author', type=int, default=None,
      help='For simplicity, the model assumes each author has the same number of items')
  parser.add_argument('--author_topic_weight', type=float, default=None,
      help='Strength of factor connecting author labels with topic proportions')
  parser.add_argument('--author_topic_iterations', type=int, default=5,
      help='Number of variational inference iterations to infer missing author labels')
  parser.add_argument('--hidden_units', type=int, nargs='+', default=None,
      help='List of hidden units defining the neural network architecture')
  parser.add_argument('--model', default=None,
      choices=list(zip(*inspect.getmembers(models, inspect.isfunction)))[0],
      help='Which model function to use')
  parser.add_argument('--model_dir', type=str,
      help='Path to trained model')
  parser.add_argument('--output_results', type=str, default='',
      help='Where to store evaluation results (JSON)')
  parser.add_argument('--output_predictions', type=str, default='',
      help='Predictions will optionally be stored in this file (CSV)')
  parser.add_argument('--embedding', default=None,
      choices=list(zip(*inspect.getmembers(embeddings, inspect.isfunction)))[0],
      help='Which embedding function to apply to data points in the training set')
  parser.add_argument('--cache', action='store_true',
      help='Cache data for faster iterations')

  args = parser.parse_args()

  # fetch parameters used during training
  args_file = os.path.join(args.model_dir, 'args.json')
  args_dict = {}

  if tf.gfile.Exists(args_file):
    with tf.gfile.GFile(args_file, 'r') as handle:
      args_dict = json.load(handle)

  for key, value in vars(args).items():
    if value is None:
      if not args_dict:
        print('Could not find hyperparameters used during training.', end=' ', file=sys.stderr)
        print('Please specify `{}` manually.'.format(key), file=sys.stderr)
        sys.exit(1)
      elif key not in args_dict:
        print('Could not find `{}` in hyperparameters.'.format(key), end=' ', file=sys.stderr)
        print('Please specify manually.', file=sys.stderr)
        sys.exit(1)
      else:
        setattr(args, key, args_dict[key])

  pprint.pprint(vars(args))

  sys.exit(main(args))
