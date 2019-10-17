"""
Downloads Pinterest data and writes it into tf-records format
"""

import json
import os
import urllib.request
from collections import defaultdict

import bson
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

CAT_PATH = 'data/pinterest/categories.txt'
BOARD_CAT_PATH = 'data/pinterest/subset_iccv_board_cate.bson'
BOARD_PIN_PATH = 'data/pinterest/subset_iccv_board_pins.bson'
PIN_IMG_PATH = 'data/pinterest/subset_iccv_pin_im.bson'

IMG_DATA_PATH = 'data/pinterest_images'
TF_DATA_PATH_TRAIN = 'data/tf_pinterest_train'
TF_DATA_PATH_TEST = 'data/tf_pinterest_test'

TARGET_IMG_SIZE = (224, 224)

if not os.path.isdir(TF_DATA_PATH_TRAIN):
    os.makedirs(TF_DATA_PATH_TRAIN)

if not os.path.isdir(TF_DATA_PATH_TEST):
    os.makedirs(TF_DATA_PATH_TEST)

if not os.path.isdir(IMG_DATA_PATH):
    os.makedirs(IMG_DATA_PATH)

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
                         'diy': [466, 29, 173, 388, 433, 25, 52, 132, 257],
                         'fitness': [448, 67, 175, 177, 211, 222, 342],
                         'fashion': [290, 305, 441, 463, 1, 162, 174, 197]}


def pad(img):
    new_im = Image.new("RGB", TARGET_IMG_SIZE)
    new_im.paste(img, ((TARGET_IMG_SIZE[0] - img.size[0]) // 2,
                       (TARGET_IMG_SIZE[1] - img.size[1]) // 2))
    return new_im


def download(board_ids, board2pin_ids, pin_id2image_url):
    print('n %s boards' % len(board_ids))
    print('n %s nominal pins' % sum([len(board2pin_ids[b]) for b in board_ids]))
    excluded_pins = []
    n_pins = 0
    for board_id in board_ids:
        for pin_id in board2pin_ids[board_id]:
            try:
                img_path = '%s/%s.jpg' % (IMG_DATA_PATH, pin_id)
                if not os.path.isfile(img_path):
                    urllib.request.urlretrieve(pin_id2image_url[pin_id], filename=img_path)
                    img = Image.open(img_path)
                    size_ratio = 1. * img.size[1] / img.size[0]
                    if 0.4 < size_ratio <= 2.:  # keep ones with good ratios only
                        img.thumbnail(TARGET_IMG_SIZE, Image.ANTIALIAS)
                        img_padded = pad(img)
                        img_padded.save(img_path)
                        print(n_pins, 'downloaded', pin_id, pin_id2image_url[pin_id], size_ratio)
                        n_pins += 1
                    else:
                        print('skipped', pin_id, pin_id2image_url[pin_id], img.size, size_ratio)
                        os.remove(img_path)
                        excluded_pins.append(pin_id)
            except:
                print('could not retrieve or load an image', pin_id, pin_id2image_url[pin_id])
                excluded_pins.append(pin_id)

    print('n boards %s' % len(board_ids))
    print('n nominal pins %s' % sum([len(board2pin_ids[b]) for b in board_ids]))
    print('n excluded pins', len(excluded_pins))
    print('n pins', n_pins)


def make_tf_record(board_ids, target, board2pin_ids, board2topic, topics):
    print('writing %s tf records' % target)
    data = []
    n_pins = 0
    for board_id in board_ids:
        for pin_id in board2pin_ids[board_id]:
            img_path = '%s/%s.jpg' % (IMG_DATA_PATH, pin_id)
            if os.path.isfile(img_path):
                try:
                    img = np.array(Image.open(img_path), dtype='uint8')
                    data.append((board_id, pin_id, img, topics[board2topic[board_id]]))
                    n_pins += 1
                except:
                    print('skipped image', img_path)

    print('n %s pins' % target, n_pins)
    tf_records_path = TF_DATA_PATH_TRAIN if target == 'train' else TF_DATA_PATH_TEST
    write_tfrecords(tf_records_path, data)
    write_meta_info(tf_records_path, data=data, target=target, topics=topics)


def write_tfrecords(target_path, data, max_items_per_tfrecord=20000):
    count = 0
    shard = 0

    writer = tf.python_io.TFRecordWriter(
        os.path.join(target_path, 'part-{0:05}.tfrecord').format(shard))

    # write TFRecord files
    for author_id, item_id, img, label in data:
        # create writers of multiple part files
        if count % max_items_per_tfrecord == 0:
            if shard > 0:
                writer.close()
            writer = tf.python_io.TFRecordWriter(
                os.path.join(target_path, 'part-{0:05}.tfrecord').format(shard))
            shard += 1

        img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2]

        # create data point
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                    'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_h])),
                    'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_w])),
                    'image_depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_c])),
                    'author_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[author_id.encode()])),
                    'author_topic': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()])),
                    'item_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item_id.encode()])),
                    'item_topic': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b''])),
                }))

        # write to training or test set
        writer.write(example.SerializeToString())
        count += 1

    writer.close()


def write_meta_info(target_path, data, target, topics):
    # write meta information into JSON files
    author_ids = list(set([entry[0] for entry in data]))

    with open(target_path.rstrip('/') + '.json', 'w') as handle:
        json.dump({
            'features': {
                'image': {'shape': [], 'dtype': 'string'},
                'image_height': {'shape': [], 'dtype': 'int64'},
                'image_width': {'shape': [], 'dtype': 'int64'},
                'image_depth': {'shape': [], 'dtype': 'int64'},
                'author_id': {'shape': [], 'dtype': 'string'},
                'author_topic': {'shape': [], 'dtype': 'string'},
                'item_id': {'shape': [], 'dtype': 'string'},
                'item_topic': {'shape': [], 'dtype': 'string'},
            },
            'meta': {
                'topics': topics,
                'author_ids': author_ids,
            }
        }, handle)


def make_new_dataset(test_size=0.5, max_boards_per_class=None, max_pins_per_board=None):
    with open(CAT_PATH) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    cat_id2name_old = dict(zip(range(1, len(content) + 1), content))
    print('n old topics', len(cat_id2name_old))

    new_cat_names = list(new_topic2old_cat_ids.keys())
    cat_name2cat_id_new = dict(zip(new_cat_names, range(len(new_cat_names))))

    print(cat_name2cat_id_new)

    cat_id_old2cat_id_new = {}
    for k, v in new_topic2old_cat_ids.items():
        for id in v:
            cat_id_old2cat_id_new[id] = cat_name2cat_id_new[k]

    print(cat_id_old2cat_id_new)

    data = bson.decode_file_iter(open(BOARD_CAT_PATH, 'rb'))
    board2cat_new = {}
    board2cat_old = {}
    board2url = {}
    for d in data:
        board2cat_new[d['board_id']] = cat_id_old2cat_id_new.get(int(d['cate_id']), -1)
        board2cat_old[d['board_id']] = int(d['cate_id'])
        board2url[d['board_id']] = d['board_url']

    cat_id2board_ids = defaultdict(list)
    for b, c in board2cat_new.items():
        if c != -1:
            cat_id2board_ids[c].append(b)

    board2pin_ids = {}
    data = bson.decode_file_iter(open(BOARD_PIN_PATH, 'rb'))
    for d in data:
        board2pin_ids[d['board_id']] = d['pins']

    board_ids, board_cats = [], []
    for k, v in cat_id2board_ids.items():
        board_ids.extend(v)
        board_cats.extend([k] * len(v))

    print('n boards', len(board_ids))
    print('n nominal pins %s' % sum([len(board2pin_ids[b]) for b in board_ids]))
    print('max pins %s' % (np.max([len(board2pin_ids[b]) for b in board_ids])))
    print('min pins %s' % (np.min([len(board2pin_ids[b]) for b in board_ids])))
    print('avg pins %s' % (np.mean([len(board2pin_ids[b]) for b in board_ids])))

    pin_id2image_url = {}
    data = bson.decode_file_iter(open(PIN_IMG_PATH, 'rb'))
    for d in data:
        pin_id2image_url[d['pin_id']] = d['im_url']

    # subsample data
    if max_boards_per_class is not None:
        board_ids, board_cats = [], []
        for k, v in cat_id2board_ids.items():
            board_ids.extend(v[:max_boards_per_class])
            board_cats.extend([k] * len(v[:max_boards_per_class]))
    if max_pins_per_board is not None:
        for k, v in board2pin_ids.items():
            board2pin_ids[k] = v[:max_pins_per_board]

    print('downloading pins')
    # download(board_ids, board2pin_ids, pin_id2image_url)

    # split into train and test sets

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for t, v in sss.split(board_ids, board_cats):
        train_idxs, test_idxs = list(t), list(v)

    # train set
    train_board_ids = [board_ids[t] for t in train_idxs]
    print('n train boards', len(train_board_ids))
    make_tf_record(train_board_ids, 'train', board2pin_ids, board2cat_new, new_cat_names)

    # test set
    test_board_ids = [board_ids[t] for t in test_idxs]
    print('n test boards', len(test_board_ids))
    make_tf_record(test_board_ids, 'test', board2pin_ids, board2cat_new, new_cat_names)


if __name__ == '__main__':
    make_new_dataset(test_size=0.2, max_boards_per_class=None, max_pins_per_board=None)
