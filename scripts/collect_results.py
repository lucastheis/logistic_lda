import json
import os
import re
from argparse import ArgumentParser

import tensorflow as tf


def print_search_results(metadata_dir, experiment_name):
    all_models = tf.gfile.ListDirectory(metadata_dir)
    exp_models = [f for f in all_models if re.match(experiment_name + r'_[0-9]+', f)]
    exp_models = [os.path.join(metadata_dir, f) for f in exp_models]

    results = []
    missing = []

    for model_dir in exp_models:
        args_file = os.path.join(model_dir, 'args.json')
        eval_file = os.path.join(model_dir, 'evaluation.txt')
        valid_file = os.path.join(model_dir, 'validation.txt')
        if not tf.gfile.Exists(args_file) or not tf.gfile.Exists(eval_file):
            missing.append(model_dir)
            continue

        try:
            res = {'id': int(model_dir.split('_')[-1])}
            with tf.gfile.GFile(args_file) as handle:
                res.update(json.load(handle))

            with tf.gfile.GFile(eval_file) as handle:
                res.update(json.load(handle))

            if tf.gfile.Exists(valid_file):
                with tf.gfile.GFile(valid_file) as handle:
                    res.update(json.load(handle))
            else:
                res.update({'accuracy_author_validation': -1.})

            results.append(res)
        except json.JSONDecodeError:
            print('Could not read the results of ', model_dir)

    results = sorted(results, key=lambda res: res['accuracy_author_validation'])
    print('n experiments', len(results))

    print('exp_id, batch_size, n_iter, alpha, regularization, lr, lr_decay, '
          'n_steps, topics_weight, n_valid, model_name, acc_valid, acc_test')

    for res in results:
        for k, v in res.items():
            if isinstance(v, (float, int)) and v < -1e+18:
                res[k] = 'None'

        print(
            '{0:>35}'.format(res['experiment_name']), '\t',
            '{0:>3}'.format(res['batch_size']), '\t',
            '{0:>3}'.format(res['n_author_topic_iterations']), '\t',
            '{0:>3}'.format(res['topic_bias_regularization']), '\t',
            '{0:>3}'.format(res['model_regularization']), '\t',
            '{0:>3}'.format(res['initial_learning_rate']), '\t',
            '{0:>3}'.format(res['learning_rate_decay']), '\t',
            '{0:>6}'.format(res['max_steps']), '\t',
            '{0:>4}'.format(res['author_topic_weight']), '\t',
            '{0:>3}'.format(res['num_valid']), '\t',
            '{0:>10}'.format(str(res['hidden_units'])), '\t',
            '{0}'.format(res['model']), '\t',
            '{0:.1f}%'.format(res['accuracy_author_validation'] * 100), '\t',
            '{0:.1f}%'.format(res['accuracy_author_test'] * 100), '\t'
        )


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--metadata_dir', type=str, default='metadata',
                        help='Path to where models are stored')
    parser.add_argument('--experiment_name', type=str, default='ng20_supervised_ce',
                        help='Name of the experiment (without a number at the end)')

    args, _ = parser.parse_known_args()
    print_search_results(args.metadata_dir, args.experiment_name)
