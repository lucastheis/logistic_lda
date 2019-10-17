"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

Script for training the models

"""

import json
import os
import pickle
import pprint
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

import data_iter_batched
import data_iter_online
import embeddings
import models
import utils

np.set_printoptions(threshold=1000)

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to where datasets are stored')
    parser.add_argument('--metadata_dir', type=str, default='metadata',
                        help='Path to where models are stored')
    parser.add_argument('--experiment_name', type=str, default='ng20_unsupervised',
                        help='Name of the experiment')
    parser.add_argument('--filename', type=str, default='tokenized_ng20_train.pkl',
                        help='Training dataset, e.g. .pkl file for Newsgroups 20 or a TFRecords file/folder')
    parser.add_argument('--output_file', type=str, default='validation.txt',
                        help='Where to write validation results')
    parser.add_argument('--vocab_file', type=str, default='vocab2K.pkl',
                        help='Vocabulary file for News20')
    parser.add_argument('--vocab_size', type=int, default=5000,
                        help='Size of the vocabulary that will be created if there is no vocabulary given')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of items or full documents per training batch')
    parser.add_argument('--author_topic_weight', type=float, default=1.0,
                        help='Strength of factor connecting author labels with topic proportions')
    parser.add_argument('--topic_bias_regularization', type=float, default=5.0,
                        help='Parameter of Dirichlet prior on topic proportions')
    parser.add_argument('--model_regularization', type=float, default=100.0,
                        help='Strength of regularization encouraging model to use many topics')
    parser.add_argument('--initial_learning_rate', type=float, default=0.0005,
                        help='Initial learning rate of optimizer')
    parser.add_argument('--learning_rate_decay', type=float, default=0.8,
                        help='Parameter of exponential learning rate decay')
    parser.add_argument('--learning_rate_decay_steps', type=int, default=2000,
                        help='Parameter of exponential learning rate decay')
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='Maximum number of updates to the model parameters')
    parser.add_argument('--num_valid', type=float, default=0,
                        help='Number of training points or proportion of traing sets used for validation')
    parser.add_argument('--items_per_author', type=int, default=200,
                        help='For simplicity, some models assume each author has the same number of items')
    parser.add_argument('--n_author_topic_iterations', type=int, default=1,
                        help='Number of variational inference iterations to infer missing author labels')
    parser.add_argument('--use_author_topics', type=int, default=0,
                        help='Supervised or unsupervised?')
    parser.add_argument('--n_unsupervised_topics', type=int, default=50,
                        help='Number of topics when trained unsupervisedly')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[128],
                        help='List of hidden units defining the neural network architecture')
    parser.add_argument('--model', type=str, default='logistic_lda',
                        choices=['logistic_lda', 'logistic_lda_ce', 'logistic_lda_online', 'mlp_online'],
                        help='Which model function to use')
    parser.add_argument('--embedding', default='glove',
                        choices=['identity', 'one_hot', 'glove', 'mobilenet_v2'],
                        help='Which embedding function to apply to data points')

    args, _ = parser.parse_known_args()

    pprint.pprint(vars(args))

    model_dir = os.path.join(args.metadata_dir, args.experiment_name)
    tf.gfile.MakeDirs(model_dir)
    with tf.gfile.GFile(os.path.join(model_dir, 'args.json'), 'w') as handle:
        json.dump(vars(args), handle, indent=4)

    if args.vocab_file:
        with tf.gfile.Open(os.path.join(args.data_dir, args.vocab_file), 'rb') as f:
            word2freq = pickle.load(f)
            word2freq = sorted(word2freq.items(), key=lambda x: x[1], reverse=True)
            words = [t[0] for t in word2freq]
            word2idx = dict(zip(words, range(len(words))))
            print('Using a vocabulary file of %s words' % len(words))
    else:
        word2idx = None

    data_iter_var = data_iter_online if 'online' in args.model else data_iter_batched

    dataset_train, dataset_valid, meta_info, _ = data_iter_var.create_tf_dataset(
        data_dir=args.data_dir,
        filename=args.filename,
        num_valid=args.num_valid,
        batch_size=args.batch_size,
        word2idx=word2idx,
        vocab_size=args.vocab_size,
        embedding=args.embedding)

    classifier = tf.estimator.Estimator(
        model_fn=getattr(models, args.model),
        model_dir=model_dir,
        params={
            'hidden_units': args.hidden_units,
            'learning_rate': args.initial_learning_rate,
            'decay_rate': args.learning_rate_decay,
            'decay_steps': args.learning_rate_decay_steps,
            'alpha': args.topic_bias_regularization,
            'model_regularization': args.model_regularization,
            'author_topic_weight': args.author_topic_weight,
            'items_per_author': args.items_per_author,
            'n_author_topic_iterations': args.n_author_topic_iterations,
            'embedding': getattr(embeddings, args.embedding),
            'use_author_topics': args.use_author_topics,
            'n_unsupervised_topics': args.n_unsupervised_topics,
            **meta_info
        })

    if 'online' in args.model:
        classifier.train(
            input_fn=lambda: dataset_train.make_one_shot_iterator().get_next(),
            max_steps=args.max_steps)
    else:
        classifier.train(
            input_fn=lambda: data_iter_batched.ConcatBatchIterator(dataset_train,
                                                                   batch_size=args.batch_size).get_next(),
            max_steps=args.max_steps)

    results = {}

    if dataset_valid is not None:
        print('Evaluating on the validation dataset')

        classifier = tf.estimator.Estimator(
            model_fn=getattr(models, args.model),
            params={
                'hidden_units': args.hidden_units,
                'learning_rate': 0.,
                'decay_rate': 0.,
                'decay_steps': 0.,
                'alpha': args.topic_bias_regularization,
                'model_regularization': 0.,
                'author_topic_weight': args.author_topic_weight,
                'items_per_author': args.items_per_author,
                'n_author_topic_iterations': args.n_author_topic_iterations,
                'embedding': getattr(embeddings, args.embedding),
                'use_author_topics': False,
                'n_unsupervised_topics': args.n_unsupervised_topics,
                **meta_info
            }, warm_start_from=model_dir)


        def get_dataset_iterator():
            return dataset_valid.make_one_shot_iterator().get_next()


        max_steps = 1 if 'ce' in args.model else None
        classifier.train(get_dataset_iterator, max_steps=max_steps)

        author_predictions = []
        author_topics = []
        author_ids = []

        for prediction in classifier.predict(get_dataset_iterator, yield_single_examples=False):
            id = [prediction['author_id']] if prediction['author_id'].shape == () else list(prediction['author_id'])
            p = [prediction['author_prediction']] if prediction['author_prediction'].shape == () else list(
                prediction['author_prediction'])

            author_predictions.extend(p)
            author_ids.extend(id)
            author_topics.extend(list(prediction['author_topic']))

        author2nitems = {}
        for id in set(author_ids):
            author2nitems[id] = author_ids.count(id)

        weights = [1.0 / author2nitems[author_id] for author_id in author_ids]
        results['accuracy_author_validation'] = utils.accuracy(author_predictions, author_topics,
                                                               weights)
        pprint.pprint(results)

    if args.output_file:
        with tf.gfile.GFile(os.path.join(model_dir, args.output_file), 'w') as handle:
            json.dump(results, handle, indent=4)
