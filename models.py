"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

Implementation of logistic LDA models.

"""

import numpy as np
import tensorflow as tf

import utils


def logistic_lda(features, labels, mode, params):
    """
     An implementation of logistic LDA, where we assume that each batch contains
     one or more full documents. Can be used as a supervised or an unsupervised model.

     Args:
       features['embedding']: A tensor of shape [B, D]
       features['author_topic']: A tensor of shape [B] containing document labels as strings
       features['author_id']: A tensor of shape [B] containing integer IDs
       features['item_topic']: A tensor of shape [B] containing item labels (use '' if unknown)
       features['item_id']: A tensor of shape [B] containing integer IDs
       labels: This will be ignored as labels are provided via `features`
       mode: Estimator's `ModeKeys`
       params['n_unsupervised_topics']: Number of topics in the case of unsupervised training
       params['topics']: A list of strings of all possible topics
       params['author_ids']: A list of all possible author IDs (these IDs group items)
       params['author2nitems']: Number of items/words per author/document
       params['hidden_units']: A list of integers describing the number of hidden units
       params['learning_rate']: Learning rate used with Adam
       params['decay_rate']: Exponential learning rate decay parameter
       params['decay_steps']: Exponential learning rate decay parameter
       params['author_topic_weight']: Controls how much author labels influence the model
       params['n_author_topic_iterations']: Number of iterations to infer missing author labels
       params['model_regularization']: Regularize model to make use of as many topics as possible
       params['alpha']: Smoothes topic distributions of authors
       params['embedding']: A function which preprocesses features
       params['use_author_topics']: If the model is unsupervised, this should be False
     """

    # compute embeddings
    features = params['embedding'](features, params)

    n_authors = len(params['author_ids'])
    n_topics = len(params['topics']) if params['n_unsupervised_topics'] <= 0 else params['n_unsupervised_topics']
    print('N topics', n_topics)
    print(params['topics'])
    n_words_total = sum(list(params['author2nitems'].values()))

    # lookup table which maps topics to indices and missing topics to -1
    topic_table = utils.create_table(
        keys=params['topics'] + [''],
        values=list(range(len(params['topics']))) + [-1])

    # convert string labels to integers
    author_topics = topic_table.lookup(features['author_topic'])
    item_topics = topic_table.lookup(features['item_topic'])

    # convert author IDs to low integers
    author_table = utils.create_table(np.asarray(params['author_ids'], dtype=np.int64))
    author_ids = author_table.lookup(features['author_id'])

    # keeps track of topic counts per user, these are bar p_d in the paper
    topic_counts_var = tf.get_variable(
        'topic_counts_var',
        shape=[n_authors, n_topics],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.),
        trainable=False,
        use_resource=True)

    # keeps track of predicted topic distributions across all items
    topic_dist_total_var = tf.get_variable(
        'topic_dist_total',
        shape=[1, n_topics],
        initializer=tf.constant_initializer(1.0 / n_topics, dtype=tf.float32),
        trainable=False,
        use_resource=True)

    # predict topics from items
    net = features['embedding']
    for units in params['hidden_units']:
        print('adding layer', units)
        net = tf.layers.dense(net, units=units,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.initializers.orthogonal,
                              bias_initializer=tf.constant_initializer(0.01))

    topic_counts = tf.gather(topic_counts_var, tf.squeeze(author_ids))

    author_topics_onehot = tf.one_hot(tf.squeeze(author_topics), n_topics)
    author_topics_prediction = tf.ones_like(author_topics_onehot) / n_topics

    # infer missing author topics
    if params['use_author_topics']:
        author_topics_prediction = tf.where(
            author_topics < 0,
            author_topics_prediction,
            author_topics_onehot)

    # update beliefs over author's topic distribution
    topic_biases = tf.digamma(params['alpha'] + topic_counts + params['author_topic_weight'] * author_topics_prediction)

    # update predictions of author topics
    author_topics_prediction = tf.nn.softmax(params['author_topic_weight'] * topic_biases)

    logits = tf.layers.dense(net, units=n_topics, activation=None)  # BxK
    logits_biased = logits + topic_biases

    # probability of each topic
    probs = tf.nn.softmax(logits)
    probs_biased = tf.nn.softmax(logits_biased)

    if mode == tf.estimator.ModeKeys.PREDICT:

        for _ in range(params['n_author_topic_iterations'] - 1):
            topic_biases = tf.digamma(
                params['alpha'] + topic_counts + params['author_topic_weight'] * author_topics_prediction)
            author_topics_prediction = tf.nn.softmax(params['author_topic_weight'] * topic_biases)
        logits_biased = logits + topic_biases
        probs_biased = tf.nn.softmax(logits_biased)

        predictions = {
            'author_id': tf.squeeze(author_ids),
            'author_prediction': tf.argmax(author_topics_prediction, -1),
            'author_probability': tf.reduce_max(author_topics_prediction, -1),
            'author_topic': author_topics,
            'item_id': features.get('item_id', tf.zeros_like(author_ids) - 1),
            'item_prediction': tf.argmax(logits_biased, 1),
            'item_probability': tf.reduce_max(probs_biased, 1),
            'item_unbiased_prediction': tf.argmax(logits, 1),
            'item_unbiased_probability': tf.reduce_max(probs, 1),
            'item_topic': item_topics,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    expected_topics = (probs + 1e-6) / (topic_dist_total_var + 1e-6) / n_topics

    # the unbiased model tries to predict the biased topics
    loss = tf.reduce_mean(
        utils.softmax_cross_entropy(
            targets=tf.stop_gradient(probs_biased + params['model_regularization'] * expected_topics),
            logits=logits))

    tf.summary.scalar('loss', loss)

    # update distribution of predicted topics
    topic_dist_diff = (probs - topic_dist_total_var) / n_words_total
    topic_dist_total_update = tf.assign_add(
        topic_dist_total_var, tf.reduce_sum(topic_dist_diff, axis=0, keepdims=True))

    # update topic counters
    reinit = tf.scatter_nd_update(topic_counts_var, author_ids, tf.zeros_like(probs_biased))
    with tf.control_dependencies([reinit]):
        topic_counts_update = tf.scatter_add(topic_counts_var, author_ids, probs_biased)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            decay_rate=params['decay_rate'],
            decay_steps=params['decay_steps'],
            global_step=tf.train.get_global_step()))
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # update model parameters and topic biases
    train_op = tf.group(train_op, topic_counts_update, topic_dist_total_update)

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)


def logistic_lda_ce(features, labels, mode, params):
    """
     An implementation of logistic LDA with cross-entropy loss when optimizing for neural network parameters.
     Here, we assume that each batch contains one or more full documents.
     Only for supervised training, i.e. when we have document labels.

     Args:
       features['embedding']: A tensor of shape [B, D]
       features['author_topic']: A tensor of shape [B] containing document labels as strings
       features['author_id']: A tensor of shape [B] containing integer IDs
       features['item_topic']: A tensor of shape [B] containing item labels (use '' if unknown)
       features['item_id']: A tensor of shape [B] containing integer IDs
       labels: This will be ignored as labels are provided via `features`
       mode: Estimator's `ModeKeys`
       params['topics']: A list of strings of all possible topics
       params['author_ids']: A list of all possible author IDs (these IDs group items)
       params['hidden_units']: A list of integers describing the number of hidden units
       params['learning_rate']: Learning rate used with Adam
       params['decay_rate']: Exponential learning rate decay parameter
       params['decay_steps']: Exponential learning rate decay parameter
       params['n_author_topic_iterations']: Number of iterations to infer missing author labels
       params['alpha']: Smoothes topic distributions of authors
       params['embedding']: A function which preprocesses features
    """
    # preprocess features (e.g., compute embeddings from words)
    features = params['embedding'](features, params)

    # lookup table which maps topics to indices and missing topics to -1
    n_topics = len(params['topics'])
    topic_table = utils.create_table(
        keys=params['topics'] + [''],
        values=list(range(n_topics)) + [-1])
    author_topics_dn = topic_table.lookup(features['author_topic'])

    # this would be a (batch_size x n_authors_in_a_batch) matrix
    unique_authors, unique_idxs = tf.unique(features['author_id'])
    n_authors = tf.shape(unique_authors)[0]
    author_1hot_mask = tf.one_hot(unique_idxs, depth=n_authors)

    author_topics_d = tf.tensordot(tf.transpose(tf.to_float(author_topics_dn)), author_1hot_mask, axes=1)
    author_topics_d = tf.transpose(author_topics_d)
    author_topics_d = author_topics_d / tf.reduce_sum(author_1hot_mask, axis=0)
    author_topics_d = tf.cast(author_topics_d, tf.int32)

    net = features['embedding']
    for units in params['hidden_units']:
        if units > 0:
            print('adding layer', units)
            net = tf.layers.dense(net, units=units,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.initializers.orthogonal,
                                  bias_initializer=tf.constant_initializer(0.01))

    logits = tf.layers.dense(net, units=n_topics, activation=None)  # BxK

    p_hat_dn = tf.ones_like(logits) / n_topics
    p_hat_d = tf.ones((n_authors, n_topics)) / n_topics

    for _ in range(params['n_author_topic_iterations']):
        sum_p_hat_dn = tf.tensordot(tf.transpose(p_hat_dn), author_1hot_mask, axes=1)
        sum_p_hat_dn = tf.transpose(sum_p_hat_dn)

        alpha_hat_d = params['alpha'] + sum_p_hat_dn + p_hat_d

        digamma_alpha_hat_d = tf.digamma(alpha_hat_d)
        digamma_alpha_hat_dn = tf.tensordot(author_1hot_mask, digamma_alpha_hat_d, axes=1)

        logits_hat_d = digamma_alpha_hat_d
        p_hat_d = tf.nn.softmax(logits_hat_d)
        p_hat_dn = tf.nn.softmax(logits + digamma_alpha_hat_dn)

    p_hat_d_expanded = tf.tensordot(author_1hot_mask, p_hat_d, axes=1)

    loss = tf.reduce_mean(
        utils.softmax_cross_entropy(
            targets=tf.one_hot(author_topics_d, depth=n_topics),
            logits=logits_hat_d))

    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'author_id': tf.squeeze(features['author_id']),
            'author_prediction': tf.argmax(p_hat_d_expanded, -1),
            'author_probability': p_hat_d_expanded,
            'author_topic': author_topics_dn,
            'item_id': features['item_id'],
            'item_prediction': tf.argmax(p_hat_dn, -1),
            'item_probability': tf.reduce_max(p_hat_dn, -1),
            'item_unbiased_prediction': tf.argmax(logits, 1),
            'item_unbiased_probability': tf.reduce_max(tf.nn.softmax(logits), 1),
            'item_topic': features['item_topic']
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            decay_rate=params['decay_rate'],
            decay_steps=params['decay_steps'],
            global_step=tf.train.get_global_step()))
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)


def logistic_lda_online(features, labels, mode, params):
    """
    An implementation of logistic LDA.
    Args:
      features['embedding']: A tensor of shape [B, D]
      features['author_topic']: A tensor of shape [B] containing author labels as strings
      features['author_id']: A tensor of shape [B] containing integer IDs
      features['item_topic']: A tensor of shape [B] containing item labels (use '' if unknown)
      features['item_id']: A tensor of shape [B] containing integer IDs
      labels: This will be ignored as labels are provided via `features`
      mode: Estimator's `ModeKeys`
      params['meta_info']['topics']: A list of strings of all possible topics
      params['meta_info']['author_ids']: A list of all possible author IDs (these IDs group items)
      params['hidden_units']: A list of integers describing the number of hidden units
      params['learning_rate']: Learning rate used with Adam
      params['decay_rate']: Exponential learning rate decay parameter
      params['decay_steps']: Exponential learning rate decay parameter
      params['author_topic_weight']: Controls how much author labels influence the model
      params['author_topic_iterations']: Number of iterations to infer missing author labels
      params['model_regularization']: Regularize model to make use of as many topics as possible
      params['items_per_author']: For simplicity, model assumes this many items per author
      params['alpha']: Smoothes topic distributions of authors
      params['embedding']: A function which preprocesses features
      params['use_author_topics']: If the model is unsupervised, this should be 0
    """

    if params['n_author_topic_iterations'] < 1:
        raise ValueError('`author_topic_iterations` should be larger than 0.')

    n_authors = len(params['author_ids'])
    n_topics = len(params['topics']) if params['n_unsupervised_topics'] <= 0 else params['n_unsupervised_topics']

    with tf.name_scope('preprocessing'):
        # lookup table which maps topics to indices and missing topics to -1
        topic_table = utils.create_table(
            keys=params['topics'] + [''],
            values=list(range(n_topics)) + [-1],
            name='topic_table')

        # convert string labels to integers
        author_topics = topic_table.lookup(features['author_topic'])
        item_topics = topic_table.lookup(features['item_topic'])

        # convert author IDs to low integers
        author_table = utils.create_table(
            keys=params['author_ids'],
            name='author_table')
        author_ids = tf.squeeze(author_table.lookup(features['author_id']))

    # preprocess features (e.g., compute embeddings from words)
    with tf.name_scope('embedding'):
        features = params['embedding'](features, params)

    # predict topics from items
    net = features['embedding']
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    with tf.name_scope('variational_inference'):
        # keeps track of topic counts per user
        topic_counts_var = tf.get_variable(
            'topic_counts',
            shape=[n_authors, n_topics],
            dtype=tf.float32,
            initializer=tf.ones_initializer,
            trainable=False,
            use_resource=True)

        # keeps track of predicted topic distributions across all items
        topic_dist_total_var = tf.get_variable(
            'topic_dist_total',
            shape=[1, n_topics],
            initializer=tf.constant_initializer(1.0 / n_topics, dtype=tf.float32),
            trainable=False,
            use_resource=True)

        # expected topic counts for each author
        topic_counts = tf.gather(topic_counts_var, author_ids)

        author_topics_onehot = tf.one_hot(tf.squeeze(author_topics), n_topics)
        author_topics_prediction = tf.ones_like(author_topics_onehot) / n_topics

        # infer missing author topics
        for _ in range(params['n_author_topic_iterations']):
            if params['use_author_topics']:
                # where available, use ground truth instead of predictions
                author_topics_prediction = tf.where(
                    author_topics < 0,
                    author_topics_prediction,
                    author_topics_onehot)

            # update beliefs over author's topic distribution
            author_alpha = params['alpha'] + topic_counts + params['author_topic_weight'] * author_topics_prediction
            topic_biases = tf.digamma(author_alpha)

            # update predictions of author topics
            author_topics_prediction = tf.nn.softmax(params['author_topic_weight'] * topic_biases)

        logits = tf.layers.dense(net, n_topics, activation=None)  # BxK
        logits_biased = logits + topic_biases

        # probability of each topic
        probs = tf.nn.softmax(logits)
        probs_biased = tf.nn.softmax(logits_biased)

        if mode == tf.estimator.ModeKeys.PREDICT:
            if params['author_topic_weight'] < 1e-8:
                author_topics_prediction = tf.nn.softmax(1e-8 * topic_biases)

            predictions = {
                'author_id': author_ids,
                'author_prediction': tf.argmax(author_topics_prediction, 1),
                'author_probability': tf.reduce_max(author_topics_prediction, 1),
                'author_topic': author_topics,
                'item_id': features.get('item_id', tf.zeros_like(author_ids) - 1),
                'item_prediction': tf.argmax(logits_biased, 1),
                'item_probability': tf.reduce_max(probs_biased, 1),
                'item_unbiased_prediction': tf.argmax(logits, 1),
                'item_unbiased_probability': tf.reduce_max(probs, 1),
                'item_topic': item_topics,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # model is regularized to predict these topics
        expected_topics = (probs + 1e-6) / (topic_dist_total_var + 1e-6) / n_topics

        # the unbiased model tries to predict the biased topics
        loss = tf.reduce_mean(
            utils.softmax_cross_entropy(
                targets=tf.stop_gradient(probs_biased + params['model_regularization'] * expected_topics),
                logits=logits))

        tf.summary.scalar('cross_entropy', loss)

        # compute upper bound on the KL divergence (up to a constant)
        with tf.name_scope('upper_bound'):
            dirichlet_entropy = tf.distributions.Dirichlet(author_alpha).entropy()
            dirichlet_entropy = tf.reduce_mean(dirichlet_entropy) / params['items_per_author']

            dirichlet_regularizer = (params['alpha'] - 1.0) * tf.reduce_sum(topic_biases, axis=1)
            dirichlet_regularizer = tf.reduce_mean(dirichlet_regularizer) / params['items_per_author']

            regularizer_entropy = tf.reduce_sum(expected_topics * tf.log(expected_topics), axis=1)
            regularizer_entropy = -tf.reduce_mean(regularizer_entropy) * params['model_regularization']

            logprobs_biased = logits_biased - tf.reduce_logsumexp(logits_biased, axis=1, keepdims=True)
            topic_entropy_plus = tf.reduce_sum(probs_biased * (logprobs_biased - topic_biases), axis=1)
            topic_entropy_plus = -tf.reduce_mean(topic_entropy_plus)

            loss = loss - tf.stop_gradient(
                dirichlet_regularizer + dirichlet_entropy + topic_entropy_plus + regularizer_entropy)

            tf.summary.scalar('upper_bound', loss)

        # update topic counters
        topic_counts_diff = probs_biased - topic_counts / params['items_per_author']
        topic_counts_update = tf.scatter_add(topic_counts_var, author_ids, topic_counts_diff)

        # update distribution of predicted topics
        topic_dist_diff = (probs - topic_dist_total_var) / (params['items_per_author'] * n_authors)
        topic_dist_total_update = tf.assign_add(
            topic_dist_total_var, tf.reduce_sum(topic_dist_diff, axis=0, keepdims=True))

        optimizer = tf.train.AdamOptimizer(
            learning_rate=tf.train.exponential_decay(
                learning_rate=params['learning_rate'],
                decay_rate=params['decay_rate'],
                decay_steps=params['decay_steps'],
                global_step=tf.train.get_global_step()))
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        # update model parameters, topic counts, and topic distribution estimate
        train_op = tf.group(train_op, topic_counts_update, topic_dist_total_update)

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)


def mlp_online(features, labels, mode, params):
    """
    A simple MLP model which can be used for topic modeling.
    We assume that every item has the same label as its document.
    Here, we assume that each batch contains random items from random documents.
    Only supervised.

    Args:
      features['embedding']: A tensor of shape [B, D]
      features['author_topic']: A tensor of shape [B] containing author labels as strings
      features['item_topic']: An tensor of shape [B] containing item labels (used in PREDICT only)
      labels: This will be ignored as labels are provided via `features`
      mode: Estimator's `ModeKeys`
      params['topics']: A list of strings of all possible topics
      params['hidden_units']: A list of integers describing the number of hidden units
      params['learning_rate']: Learning rate used with Adam
      params['decay_rate']: Exponential learning rate decay parameter
      params['decay_steps']: Exponential learning rate decay parameter
      params['embedding']: A function which preprocesses features

    Returns:
      A `tf.estimator.EstimatorSpec`
    """

    n_topics = len(params['topics'])

    # preprocess features (e.g., compute embeddings from words)
    features = params['embedding'](features, params)

    # convert string labels to integers
    topic_table = utils.create_table(params['topics'])
    author_topics = topic_table.lookup(features['author_topic'])

    # convert author IDs to low integers
    author_table = utils.create_table(params['author_ids'])
    author_ids = author_table.lookup(features['author_id'])

    net = features['embedding']
    for units in params['hidden_units']:
        net = tf.layers.dense(
            net,
            units=units,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(params['model_regularization']))

    logits = tf.layers.dense(
        net,
        n_topics,
        activation=None,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(params['model_regularization']))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'author_id': tf.squeeze(author_ids),
            'author_prediction': tf.argmax(logits, 1),
            'author_topic': author_topics,
            'item_id': features['item_id'],
            'item_prediction': tf.argmax(logits, 1),
            'item_probability': tf.reduce_max(tf.nn.softmax(logits), 1),
            'item_unbiased_prediction': tf.argmax(logits, 1),
            'item_topic': topic_table.lookup(features['item_topic'])
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # model is trained to predict which topic an author belongs to
    loss = tf.reduce_mean(
        utils.softmax_cross_entropy(
            targets=tf.one_hot(author_topics, depth=n_topics),
            logits=logits))

    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            decay_rate=params['decay_rate'],
            decay_steps=params['decay_steps'],
            global_step=tf.train.get_global_step()))
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)
