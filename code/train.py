"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

Train either logistic LDA or a simple MLP to predict topics from tweets.
"""

import json
import os
import pprint
from argparse import ArgumentParser
from functools import partial

from logistic_lda.data import create_datasets
from logistic_lda import embeddings, models

import tensorflow as tf


def main(args):
  # prepare training and validation data
  if args.num_valid > 0:
    dataset_train, dataset_valid, meta_info = create_datasets(**vars(args))
  else:
    dataset_train, meta_info = create_datasets(**vars(args))

  classifier = tf.estimator.Estimator(
    model_fn=getattr(models, args.model + '_fn'),
    params={
      'hidden_units': args.hidden_units,
      'learning_rate': args.initial_learning_rate,
      'decay_rate': args.learning_rate_decay,
      'decay_steps': args.learning_rate_decay_steps,
      'alpha': args.topic_bias_regularization,
      'model_regularization': args.model_regularization,
      'max_items_per_author': args.max_items_per_author,
      'author_topic_weight': args.author_topic_weight,
      'author_topic_iterations': args.author_topic_iterations,
      'meta_info': meta_info,
      'embedding': partial(getattr(embeddings, args.embedding), meta_info=meta_info, args=args, mode='train'),
      'use_author_topics': True,
    },
    model_dir=args.model_dir,
    warm_start_from=args.warm_start_from)

  # train
  classifier.train(
    input_fn=lambda: dataset_train.make_one_shot_iterator().get_next(),
    max_steps=args.max_steps)

  # evaluate
  if args.num_valid > 0:
    classifier.evaluate(lambda: dataset_valid.make_one_shot_iterator().get_next())

  return 0


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', type=str,
      default='/Users/ltheis/Projects/topic_model/data/tweet_embeddings_train/')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--author_topic_weight', type=float, default=200)
  parser.add_argument('--author_topic_iterations', type=float, default=1)
  parser.add_argument('--topic_bias_regularization', type=float, default=0.5)
  parser.add_argument('--model_regularization', type=float, default=2.5)
  parser.add_argument('--initial_learning_rate', type=float, default=0.001)
  parser.add_argument('--learning_rate_decay', type=float, default=0.8)
  parser.add_argument('--learning_rate_decay_steps', type=int, default=2000)
  parser.add_argument('--max_epochs', type=int, default=200)
  parser.add_argument('--max_steps', type=int, default=3000)
  parser.add_argument('--max_items_per_author', type=int, default=200)
  parser.add_argument('--num_valid', type=int, default=0)
  parser.add_argument('--hidden_units', type=int, nargs='*', default=[512, 256, 128])
  parser.add_argument('--model', default='logistic_lda', choices=['mlp', 'logistic_lda', 'logistic_lda5'])
  parser.add_argument('--embedding', default='identity', choices=['identity', 'one_hot', 'weighted', 'universal_sentence_enc', 'bilstm'])
  parser.add_argument('--embedding_path', default=None, type=str)
  parser.add_argument('--embedding_vocab_size', default=None, type=int)
  parser.add_argument('--embedding_dim', default=None, type=int)
  parser.add_argument('--attention_path', default=None, type=str)
  parser.add_argument('--model_dir', type=str,
      default='/Users/ltheis/Projects/topic_model/models/0')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--warm_start_from', type=str, default=None)

  args, _ = parser.parse_known_args()

  pprint.pprint(vars(args))

  if args.overwrite and tf.gfile.Exists(args.model_dir):
    tf.gfile.DeleteRecursively(args.model_dir)

  tf.gfile.MkDir(args.model_dir)
  with tf.gfile.GFile(os.path.join(args.model_dir, 'args.json'), 'w') as handle:
    json.dump(vars(args), handle, indent=4)

  tf.logging.set_verbosity(tf.logging.INFO)

  main(args)
