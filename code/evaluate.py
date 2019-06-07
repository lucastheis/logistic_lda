"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

Evaluates accuracy with which a model predicts labels, as well as mutual information between
predicted topics and labels (which can be used to evaluate unsupervised models).
"""

from __future__ import print_function

import csv
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
from scipy.optimize import linear_sum_assignment


def mutual_information(part0, part1, weights=None):
  """
  Estimates from observations how much information one categorical random variable contains about
  another.

  Args:
    part0: A `list[int]` of labels
    part1: A `list[int]` of predicted labels

  Returns:
    A `float` representing the mutual information between the two variables
  """

  # remove unlabeled data
  part0, part1 = _clean(part0, part1)

  if len(part0) == 0:
    return 0.0

  max0 = np.max(part0)
  max1 = np.max(part1)

  if weights is None:
    weights = np.ones_like(part0)

  prob = np.zeros([max0 + 1, max1 + 1])
  for i, j, w in zip(part0, part1, weights):
    prob[i, j] += w
  prob /= np.sum(weights)

  prob0 = np.sum(prob, axis=1, keepdims=True)
  prob1 = np.sum(prob, axis=0, keepdims=True)

  ratio = np.divide(prob, prob0 * prob1, where=(prob > 0.0), out=np.ones_like(prob))
  return np.sum(prob * np.log(ratio))


def accuracy(preds, labels, weights=None):
  if weights is None:
    weights = np.ones_like(preds)

  # remove unlabeled data
  preds, labels, weights = _clean(preds, labels, weights)

  if len(labels) == 0:
    return 0.0

  return np.sum([w * (p == l) for w, p, l in zip(weights, preds, labels)]) / float(np.sum(weights))


def accuracy_remapped(predictions, labels, weights=None):
  if weights is None:
    weights = np.ones_like(labels)

  K = max(max(labels), max(predictions)) + 1

  CM = np.zeros([K, K], dtype=float)
  for p, l, w in zip(predictions, labels, weights):
    CM[p, l] += w

  rows, cols = linear_sum_assignment(-CM)

  return np.sum(CM[rows, cols]) / np.sum(weights)


def _clean(*lists):
  """
  Removes elements from both lists if element in one list is negative.
  """

  def _all_non_neg(values):
    return np.all(np.asarray(values) >= 0)

  lists_clean = list(zip(*filter(_all_non_neg, zip(*lists))))

  if len(lists_clean) > 1:
    return lists_clean

  return [[] for _ in lists]


def main(args):
  # prepare training and validation data
  dataset_test, meta_info = create_datasets(max_epochs=args.num_iter, **vars(args))

  def get_dataset_iterator():
    return dataset_test.make_one_shot_iterator().get_next()

  embedding = partial(getattr(embeddings, args.embedding), meta_info=meta_info, args=args, mode='valid')

  # infer topics and author's topic distributions (initialization)
  classifier = tf.estimator.Estimator(
      model_fn=getattr(models, args.model + '_fn'),
      params={
        'hidden_units': args.hidden_units,
        'learning_rate': 0.0,
        'decay_rate': 0.0,
        'decay_steps': 0,
        'alpha': args.topic_bias_regularization,
        'model_regularization': 0.0,
        'author_topic_weight': 0.0,  # this avoids local optima
        'author_topic_iterations': 1,
        'max_items_per_author': args.max_items_per_author,
        'meta_info': meta_info,
        'embedding': embedding,
        'use_author_topics': False,
      },
      warm_start_from=args.model_dir,
    )
  classifier.train(get_dataset_iterator)

  # infer topics and author's topic distributions
  classifier = tf.estimator.Estimator(
      model_fn=getattr(models, args.model + '_fn'),
      params={
        'hidden_units': args.hidden_units,
        'learning_rate': 0.0,
        'decay_rate': 0.0,
        'decay_steps': 0,
        'alpha': args.topic_bias_regularization,
        'model_regularization': 0.0,
        'author_topic_weight': args.author_topic_weight,
        'author_topic_iterations': args.author_topic_iterations,
        'max_items_per_author': args.max_items_per_author,
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

  # evaluate mutual information
  results['entropy_author'] = mutual_information(author_topics, author_topics)
  results['mi_author'] = mutual_information(author_predictions, author_topics)
  results['mi_author_relative'] = float(np.array(results['mi_author']) / results['entropy_author'])

  results['entropy_item'] = mutual_information(item_topics, item_topics)
  results['mi_item'] = mutual_information(item_predictions, item_topics)
  results['mi_item_relative'] = float(np.array(results['mi_item']) / results['entropy_item'])

  # evaluate accuracy
  results['accuracy_author'] = accuracy(author_predictions, author_topics)
  results['accuracy_author_unweighted'] = accuracy(author_predictions, author_topics, weights)
  results['accuracy_author_remapped'] = accuracy_remapped(author_predictions, author_topics, weights)
  results['accuracy_item'] = accuracy(item_predictions, item_topics)
  results['accuracy_item_remapped'] = accuracy_remapped(item_predictions, item_topics)

  # evaluate confusion matrices
  cm_u = np.zeros([len(meta_info['topics']), len(meta_info['topics'])], dtype=int)
  for t, p in zip(author_topics, author_predictions):
    cm_u[t][p] += 1

  cm_i = np.zeros_like(cm_u)
  for t, p in zip(item_topics, item_predictions):
    cm_i[t][p] += 1

  # evaluate calibrated accuracy
  results['accuracy_author_calibrated'] = np.sum(np.max(cm_u, axis=0)) / float(np.sum(cm_u))
  results['accuracy_item_calibrated'] = np.sum(np.max(cm_i, axis=0)) / float(np.sum(cm_i))

  pprint.pprint(results)

  results['confusion_matrix_author'] = cm_u.tolist()
  results['confusion_matrix_item'] = cm_i.tolist()

  if args.output:
    with tf.gfile.GFile(args.output, 'w') as handle:
      json.dump(results, handle, indent=4)

  if args.output_predictions:
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


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', type=str,
      default='/Users/ltheis/Projects/topic_model/data/tweet_embeddings_test')
  parser.add_argument('--batch_size', type=int, default=None)
  parser.add_argument('--num_iter', type=int, default=1)
  parser.add_argument('--topic_bias_regularization', type=float, default=None)
  parser.add_argument('--max_items_per_author', type=int, default=None)
  parser.add_argument('--hidden_units', type=int, nargs='+', default=None)
  parser.add_argument('--model', default=None, choices=['mlp', 'logistic_lda', 'logistic_lda5'])
  parser.add_argument('--model_dir', type=str,
      default='/Users/ltheis/Projects/topic_model/models/0')
  parser.add_argument('--author_topic_weight', type=float, default=None)
  parser.add_argument('--author_topic_iterations', type=int, default=5)
  parser.add_argument('--output', type=str, default='')
  parser.add_argument('--output_predictions', type=str, default='',
      help='Predictions will optionally be stored in this file in CSV format')
  parser.add_argument('--cache', action='store_true', help='Cache data for faster iterations')
  parser.add_argument('--embedding', default=None, choices=['identity', 'one_hot', 'weighted', 'bilstm'])
  parser.add_argument('--embedding_path', default=None, type=str)
  parser.add_argument('--embedding_vocab_size', default=None, type=int)
  parser.add_argument('--embedding_dim', default=None, type=int)
  parser.add_argument('--attention_path', default=None, type=str)

  args = parser.parse_args()

  # stores parameters used during training
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
