"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

This module implements embeddings so they can be reused by different models.

"""

import tensorflow as tf
import tensorflow_hub as hub

import data_utils


def identity(features, meta_info):
    """
    Assumes embeddings are already present in the data.
    """
    return features


def one_hot(features, meta_info):
    """
    Assumes data contains word indices. Returns one-hot encoded words.
    """
    num_words = len(meta_info['word2idx'])
    features['embedding'] = tf.one_hot(features['embedding'], depth=num_words)
    return features


def glove(features, meta_info):
    """
    Assumes data contains word indices. Returns glove word embeddings.
    """
    word2embedding = data_utils.load_glove_embeddings(meta_info['data_dir'])
    embedding_matrix = data_utils.create_embedding_matrix(word2embedding, meta_info['word2idx'])
    embedding_matrix_tf = tf.get_variable('embedding_matrix', initializer=tf.constant(embedding_matrix),
                                          trainable=False)
    features['embedding'] = tf.nn.embedding_lookup(embedding_matrix_tf, features['embedding'])
    return features


def mobilenet_v2(features, meta_info):
    remote_model_path = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"
    module_spec = hub.load_module_spec(remote_model_path)
    images = tf.cast(features['embedding'], dtype=tf.float32) / 255.
    module = hub.Module(module_spec, trainable=False)
    features['embedding'] = module(images)
    return features
