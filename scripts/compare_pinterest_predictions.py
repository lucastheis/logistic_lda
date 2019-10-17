"""
Implements majority voting for MLP predictions and compares to the performance of logistic LDA

"""
import csv
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

new_topic2old_cat_ids = {'food': [275, 262, 363, 366, 383, 411, 449, 458, 468, 451, 204, 57],
                         'animals': [249, 327, 370, 381, 419, 439, 206],
                         'women_fashion': [224, 282, 87, 452],
                         'men_fashion': [176, 169, 190, 217, 248, 265, 368],
                         'garden': [306, 447, 2, 71, 117],
                         'architecture': [427, 22, 236],
                         'wedding': [252, 255, 393, 432, 5, 120, 159],
                         'cars': [322, 84],
                         'hair': [450, 467, 30, 40, 106, 148, 287],
                         'home': [108, 109, 111, 218, 279],
                         'travel': [465, 20, 125, 238, 272, 325, 425],
                         'DIY': [466, 29, 173, 388, 433, 25, 52, 132, 257],
                         'fitness': [448, 67, 175, 177, 211, 222, 342],
                         'fashion': [290, 305, 441, 463, 1, 162, 174, 197]}

print(new_topic2old_cat_ids.keys())

new_cat_names = list(new_topic2old_cat_ids.keys())
cat_name2cat_id_new = dict(zip(new_cat_names, range(len(new_cat_names))))
cat_id2cat_name = dict(zip(range(len(new_cat_names)), new_cat_names))


def majority_vote(x):
    return max(set(x), key=x.count)


def read_csv_into_dicts(csv_path):
    board2pins = defaultdict(list)
    board2prediction = defaultdict(list)
    board2label = defaultdict(list)
    with tf.gfile.GFile(csv_path) as handle:
        csv_reader = csv.reader(handle)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names:', ' '.join(row))
                line_count += 1
            else:
                board_id = row[0]
                board2label[board_id].append(int(row[1]))
                board2prediction[board_id].append(int(row[2]))
                item_id = row[3].replace('b\'', '').replace('\'', '') + '.jpg'
                board2pins[board_id].append((item_id, int(row[5]), int(row[6]), float(row[7])))
    return board2pins, board2prediction, board2label


def accuracy_board(board2pred, board2label):
    y_true = []
    y_pred = []
    for b in board2pred.keys():
        label = majority_vote(board2label[b])
        pred = majority_vote(board2pred[b])
        y_true.append(label)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    print('n wrong items', sum((np.array(y_true) - np.array(y_pred)) != 0))
    print('acc %s_boards:' % len(y_true), acc)


def accuracy_from_1item(board2pins, board2label):
    y_true = []
    y_pred = []
    for b in board2pins.keys():
        preds = [l[2] for l in board2pins[b]]
        labels = board2label[b]
        assert len(preds) == len(labels)
        y_true.extend(labels)
        y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    print('acc %s_items: board label from 1 item' % len(y_true), acc)


def accuracy_from_1_item_weighted(board2pins, board2label):
    y_true = []
    y_pred = []
    weights = []
    for b in board2pins.keys():
        preds = [l[2] for l in board2pins[b]]
        labels = board2label[b]
        assert len(preds) == len(labels)
        y_true.extend(labels)
        y_pred.extend(preds)
        weights.extend([1. / len(preds)] * len(preds))
    acc = accuracy_score(y_true, y_pred, sample_weight=weights)
    print('acc %s_items: board label from 1 item weighted' % len(y_true), acc)


def accuracy_from_all_items_lda(board2pins, board2label, board2prediction):
    y_true = []
    y_pred = []
    for b in board2pins.keys():
        preds = board2prediction[b]
        labels = board2label[b]
        assert len(preds) == len(labels)
        y_true.extend(labels)
        y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    print('n wrong items', sum((np.array(y_true) - np.array(y_pred)) != 0))
    print('acc %s_items: board label from all items' % len(y_true), acc)


def accuracy_from_all_items_mlp(board2pins, board2label):
    y_true = []
    y_pred = []
    for b in board2pins.keys():
        labels = board2label[b]
        preds = [majority_vote([l[1] for l in board2pins[b]])] * len(labels)
        assert len(preds) == len(labels)
        y_true.extend(labels)
        y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    print('n wrong items', sum((np.array(y_true) - np.array(y_pred)) != 0))
    print('acc %s_items: board from all items' % len(y_true), acc)


def evaluate(lda_predictions_path, mlp_predictions_path):
    board2pins_lda, board2prediction_lda, board2label_lda = read_csv_into_dicts(lda_predictions_path)
    board2pins_mlp, board2prediction_mlp, board2label_mlp = read_csv_into_dicts(mlp_predictions_path)

    print('LDA:')
    accuracy_board(board2prediction_lda, board2label_lda)
    accuracy_from_1item(board2pins_lda, board2label_lda)
    accuracy_from_all_items_lda(board2pins_lda, board2label_lda, board2prediction_lda)
    accuracy_from_1_item_weighted(board2pins_lda, board2label_lda)

    print('MLP:')
    accuracy_board(board2prediction_mlp, board2label_mlp)
    accuracy_from_1item(board2pins_mlp, board2label_mlp)
    accuracy_from_all_items_mlp(board2pins_mlp, board2label_mlp)
    accuracy_from_1_item_weighted(board2pins_mlp, board2label_mlp)

    print('n boards', len(board2pins_lda))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--metadata_dir', type=str, default='metadata',
                        help='Path to where models are stored')
    parser.add_argument('--exp_name_lda', type=str, default='pinterest_supervised',
                        help='Name of the experiment')
    parser.add_argument('--exp_name_mlp', type=str, default='pinterest_mlp',
                        help='Name of the experiment')

    args, _ = parser.parse_known_args()
    evaluate(os.path.join(args.metadata_dir, args.exp_name_lda, 'predictions.csv'),
             os.path.join(args.metadata_dir, args.exp_name_mlp, 'predictions.csv'))
