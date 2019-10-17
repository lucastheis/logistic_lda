"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

Evaluation script for both supervised and unsupervised models.

"""

import csv
import json
import os
import pickle
import pprint
import sys
from argparse import ArgumentParser
from collections import defaultdict

import tensorflow as tf
from PIL import Image, ImageOps

import data_iter_batched
import data_iter_online
import embeddings
import models
import utils


def evaluate_unsupervised_ng20(dataset, meta_info, args):
    def get_dataset_iterator():
        return dataset.make_one_shot_iterator().get_next()

    classifier = tf.estimator.Estimator(
        model_fn=getattr(models, args.model),
        params={
            'hidden_units': args.hidden_units,
            'learning_rate': 0.0,
            'decay_rate': 0.0,
            'decay_steps': 0,
            'model_regularization': 0.0,
            'alpha': args.topic_bias_regularization,
            'author_topic_weight': args.author_topic_weight,
            'items_per_author': args.items_per_author,
            'n_author_topic_iterations': args.n_author_topic_iterations,
            'embedding': getattr(embeddings, args.embedding),
            'use_author_topics': False,
            'n_unsupervised_topics': args.n_unsupervised_topics,
            **meta_info
        },
        warm_start_from=model_dir
    )

    classifier.train(get_dataset_iterator, max_steps=1)

    word2idx = meta_info['word2idx']
    idx2word = {y: x for x, y in word2idx.items()}
    topic2word_prob = defaultdict(list)
    words, preds, probs = [], [], []

    for prediction in classifier.predict(get_dataset_iterator, yield_single_examples=False):
        item_id = list(prediction['item_id'])[0]
        item_topic_pred = list(prediction['item_prediction'])[0]
        item_topic_prob = prediction['item_probability'][0]
        word = idx2word[item_id]

        words.append(word)
        preds.append(item_topic_pred)
        probs.append(item_topic_prob)

        topic2word_prob[item_topic_pred].append((word, item_topic_prob))

    print('Writing predictions into', args.output_predictions_file)
    with tf.gfile.GFile(os.path.join(model_dir, args.output_predictions_file), 'w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['item', 'item_topic', 'topic_probability'])
        for i in range(len(words)):
            writer.writerow([
                words[i],
                preds[i],
                probs[i]])

    print('\n ------- Top 15 words from every topic ---------')
    for topic in topic2word_prob.keys():
        values = topic2word_prob[topic]
        sorted_by_prob = sorted(values, key=lambda tup: tup[1], reverse=True)[:15]
        words_only = ' '.join([t[0] for t in sorted_by_prob])
        print(words_only)

    return 0


def evaluate_unsupervised_pinterest(dataset, meta_info, args):
    def get_dataset_iterator():
        return dataset.make_one_shot_iterator().get_next()

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
            'items_per_author': args.items_per_author,
            'n_author_topic_iterations': args.n_author_topic_iterations,
            'embedding': getattr(embeddings, args.embedding),
            'use_author_topics': False,
            'n_unsupervised_topics': args.n_unsupervised_topics,
            **meta_info
        },
        warm_start_from=model_dir
    )

    # if stochastic VI, we need training steps to update the statistics, othw we don't
    max_steps = 1 if '_ce' in args.model else None
    classifier.train(get_dataset_iterator, max_steps=max_steps)

    author_predictions, author_topics, author_ids = [], [], []
    item_predictions, item_unbiased_predictions, item_probabilities, item_unbiased_probabilities, item_topics, item_ids = [], [], [], [], [], []

    # get labels and predictions
    for prediction in classifier.predict(get_dataset_iterator, yield_single_examples=False):
        id = [prediction['author_id']] if prediction['author_id'].shape == () else list(prediction['author_id'])
        p = [prediction['author_prediction']] if prediction['author_prediction'].shape == () else list(
            prediction['author_prediction'])

        author_predictions.extend(p)
        author_ids.extend(id)
        author_topics.extend(list(prediction['author_topic']))
        item_predictions.extend(list(prediction['item_prediction']))
        item_unbiased_predictions.extend(list(prediction.get('item_unbiased_prediction', [-1] * len(id))))
        item_probabilities.extend(list(prediction['item_probability']))
        item_unbiased_probabilities.extend(list(prediction['item_unbiased_probability']))
        item_topics.extend(list(prediction['item_topic']))
        item_ids.extend(list(prediction['item_id']))

    item2pred, item2prob = {}, {}
    for i in range(len(item_ids)):
        item2pred[item_ids[i]] = item_unbiased_predictions[i]
        item2prob[item_ids[i]] = item_unbiased_probabilities[i]

    topic2word_prob = defaultdict(list)
    for k, v in item2pred.items():
        topic2word_prob[v].append((k, item2prob[k]))

    for topic in topic2word_prob.keys():
        values = topic2word_prob[topic]
        sorted_by_prob = sorted(values, key=lambda tup: tup[1], reverse=True)
        print(topic, sorted_by_prob[:20])

    MAX_IMAGES = 20
    IMG_W = 224
    BORDER_W = 2
    IMG_NEW_W = 2 * BORDER_W + IMG_W
    TEXT_H, TEXT_W = 0, 0

    for topic in topic2word_prob.keys():
        print(topic)
        values = topic2word_prob[topic]
        sorted_by_prob = sorted(values, key=lambda tup: tup[1], reverse=True)
        ids_only = [t[0] for t in sorted_by_prob][:MAX_IMAGES]
        path_str = []
        for id in ids_only:
            img_name = str(id).replace('b\'', '').replace('\'', '') + '.jpg'
            img_path = os.path.join(args.data_dir, 'pinterest_images', img_name)
            path_str.append(img_path)

        print(path_str)

        W = IMG_NEW_W * len(path_str)
        new_im = Image.new('RGB', (W + TEXT_W, IMG_NEW_W + TEXT_H), color='white')

        i = 0
        for j in range(TEXT_W, W, IMG_NEW_W):
            im = Image.open(path_str[i])
            im = ImageOps.expand(im, border=BORDER_W, fill='white')
            new_im.paste(im, (j, 10))
            i += 1
        img_save_path = os.path.join(model_dir, '%s_tile.png' % int(topic))
        new_im.save(img_save_path)
        print('image saved in', img_save_path)

    if args.output_predictions_file:
        print('writing predictions into a csv file')
        with tf.gfile.GFile(args.output_predictions_file, 'w') as handle:
            writer = csv.writer(handle)
            writer.writerow([
                'author_id',
                'author_topic',
                'author_prediction',
                'item_id',
                'item_topic',
                'item_prediction',
                'item_unbiased_prediction',
                'item_probability',
                'item_unbiased_probability'])

            for i in range(len(author_ids)):
                writer.writerow([
                    author_ids[i],
                    author_topics[i],
                    author_predictions[i],
                    item_ids[i],
                    item_topics[i],
                    item_predictions[i],
                    item_unbiased_predictions[i],
                    item_probabilities[i],
                    item_unbiased_probabilities[i]])

    return 0


def evaluate_supervised(dataset, meta_info, args, results, subset):
    def get_dataset_iterator():
        return dataset.make_one_shot_iterator().get_next()

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
            'items_per_author': args.items_per_author,
            'n_author_topic_iterations': args.n_author_topic_iterations,
            'embedding': getattr(embeddings, args.embedding),
            'use_author_topics': False,
            'n_unsupervised_topics': args.n_unsupervised_topics,
            **meta_info
        },
        warm_start_from=model_dir
    )

    # if stochastic VI, we need training steps to update the statistics, otherwise we don't
    max_steps = 1 if '_ce' in args.model else None
    classifier.train(get_dataset_iterator, max_steps=max_steps)

    author_predictions, author_topics, author_ids = [], [], []
    item_predictions, item_unbiased_predictions, item_probabilities, item_unbiased_probabilities, item_topics, item_ids = [], [], [], [], [], []

    # get labels and predictions
    for prediction in classifier.predict(get_dataset_iterator, yield_single_examples=False):
        id = [prediction['author_id']] if prediction['author_id'].shape == () else list(prediction['author_id'])
        p = [prediction['author_prediction']] if prediction['author_prediction'].shape == () else list(
            prediction['author_prediction'])

        author_predictions.extend(p)
        author_ids.extend(id)
        author_topics.extend(list(prediction['author_topic']))
        item_predictions.extend(list(prediction['item_prediction']))
        item_probabilities.extend(list(prediction['item_probability']))
        item_topics.extend(list(prediction['item_topic']))
        item_ids.extend(list(prediction['item_id']))
        item_unbiased_predictions.extend(list(prediction.get('item_unbiased_prediction', [-1] * len(id))))
        item_unbiased_probabilities.extend(list(prediction.get('item_unbiased_probability', [-1] * len(id))))

    item2pred, item2prob = {}, {}
    for i in range(len(item_ids)):
        item2pred[item_ids[i]] = item_predictions[i]
        item2prob[item_ids[i]] = item_probabilities[i]

    topic2word_prob = defaultdict(list)
    for k, v in item2pred.items():
        topic2word_prob[v].append((k, item2prob[k]))

    author2nitems = {}
    for id in set(author_ids):
        author2nitems[id] = author_ids.count(id)

    weights = [1.0 / author2nitems[author_id] for author_id in author_ids]
    results['accuracy_author_%s' % subset] = utils.accuracy(author_predictions, author_topics,
                                                            weights)

    # author prediction is repeated n_items times
    results['accuracy_author_weighted_%s' % subset] = utils.accuracy(author_predictions, author_topics)

    pprint.pprint(results)

    print('writing predictions into a csv file')
    if args.output_predictions_file:
        with tf.gfile.GFile(os.path.join(model_dir, args.output_predictions_file), 'w') as handle:
            writer = csv.writer(handle)
            writer.writerow([
                'author_id',
                'author_topic',
                'author_prediction',
                'item_id',
                'item_topic',
                'item_prediction',
                'item_unbiased_prediction',
                'item_probability',
                'item_unbiased_probability'])

            for i in range(len(author_ids)):
                writer.writerow([
                    author_ids[i],
                    author_topics[i],
                    author_predictions[i],
                    item_ids[i],
                    item_topics[i],
                    item_predictions[i],
                    item_unbiased_predictions[i],
                    item_probabilities[i],
                    item_unbiased_probabilities[i]])

    return 0


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--experiment_name', type=str, default='ng20_unsupervised_debug')
    parser.add_argument('--evaluate_unsupervised', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--metadata_dir', type=str, default='metadata')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--hidden_units', type=int, nargs='+', default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--topic_bias_regularization', type=float, default=None)
    parser.add_argument('--author_topic_weight', type=float, default=None)
    parser.add_argument('--model_regularization', type=float, default=None)
    parser.add_argument('--num_valid', type=float, default=None)
    parser.add_argument('--items_per_author', type=int, default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--n_unsupervised_topics', type=int, default=None)
    parser.add_argument('--n_author_topic_iterations', type=int, default=None)
    parser.add_argument('--embedding', type=str, default=None)
    parser.add_argument('--use_author_topics', type=int, default=None)
    parser.add_argument('--output_predictions_file', type=str, default='predictions.csv')
    parser.add_argument('--cache', action='store_true', help='Cache data for faster iterations')

    args = parser.parse_args()

    model_dir = os.path.join(args.metadata_dir, args.experiment_name)

    # stores parameters used during training
    args_file = os.path.join(model_dir, 'args.json')
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

    if args.vocab_file:
        with tf.gfile.Open(os.path.join(args.data_dir, args.vocab_file), 'rb') as f:
            word2freq = pickle.load(f)
            word2freq = sorted(word2freq.items(), key=lambda x: x[1], reverse=True)
            words = [t[0] for t in word2freq]
            word2idx = dict(zip(words, range(len(words))))
    else:
        word2idx = None

    filename_train = args.filename
    filename_test = args.filename.replace('train', 'test')

    data_iter_var = data_iter_online if 'online' in args.model else data_iter_batched

    if not args.evaluate_unsupervised:
        results = {}
        output_file = os.path.join(model_dir, 'evaluation.txt')

        dataset_train, _, meta_info_train, word2idx = data_iter_var.create_tf_dataset(data_dir=args.data_dir,
                                                                                      filename=filename_train,
                                                                                      vocab_size=args.vocab_size,
                                                                                      max_epochs=1,
                                                                                      batch_size=args.batch_size,
                                                                                      num_valid=args.num_valid,
                                                                                      embedding=args.embedding,
                                                                                      word2idx=word2idx)
        print('Evaluating on the test set')
        dataset_test, _, meta_info_test, _ = data_iter_var.create_tf_dataset(data_dir=args.data_dir,
                                                                             filename=filename_test,
                                                                             word2idx=word2idx,
                                                                             batch_size=args.batch_size,
                                                                             max_epochs=1,
                                                                             num_valid=0,
                                                                             embedding=args.embedding)
        evaluate_supervised(dataset_test, meta_info_test, args, results, subset='test')

        with tf.gfile.GFile(output_file, 'w') as handle:
            json.dump(results, handle, indent=4)

    else:
        if 'ng20' in filename_train:
            print('Computing topics for vocabulary words')
            dataset_vocab, meta_info_vocab, _ = data_iter_var.create_tf_vocab_dataset(word2idx,
                                                                                      data_dir=args.data_dir)
            evaluate_unsupervised_ng20(dataset_vocab, meta_info_vocab, args)
        elif 'pinterest' in filename_train:
            dataset_train, _, meta_info_train, word2idx = data_iter_var.create_tf_dataset(data_dir=args.data_dir,
                                                                                          filename=filename_train,
                                                                                          vocab_size=args.vocab_size,
                                                                                          max_epochs=1,
                                                                                          batch_size=args.batch_size,
                                                                                          num_valid=args.num_valid,
                                                                                          embedding=args.embedding,
                                                                                          word2idx=word2idx)
            dataset_test, _, meta_info_test, _ = data_iter_var.create_tf_dataset(data_dir=args.data_dir,
                                                                                 filename=filename_test,
                                                                                 word2idx=word2idx,
                                                                                 batch_size=args.batch_size,
                                                                                 max_epochs=1,
                                                                                 num_valid=0,
                                                                                 embedding=args.embedding)
            evaluate_unsupervised_pinterest(dataset_test, meta_info_test, args)
        else:
            raise ValueError('no evaluation method for this dataset')
