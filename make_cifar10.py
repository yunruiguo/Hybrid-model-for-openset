#!/usr/bin/env python
# Downloads the CIFAR-10 image dataset
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import pickle

DATA_DIR = '../data'
DATASET_NAME = 'cifar10'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

IMAGES_LABELS_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']



def main():

    print("{} dataset download script initializing...".format(DATASET_NAME))
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    mkdir(DATASET_PATH + '/cifar-10-batches-py/')
    PNG_PATH = DATASET_PATH + '/images/'
    mkdir(PNG_PATH)
    mkdir(DATASET_PATH + '/images/train/')
    mkdir(DATASET_PATH + '/images/test/')

    print("Downloading {} dataset files to {}...".format(DATASET_NAME, DATASET_PATH))
    download(DATASET_PATH, 'cifar-10-python.tar.gz', IMAGES_LABELS_URL)

    train_labels = []
    train_filenames = []
    train_data = []
    for i in range(1, 6):
        data_file = os.path.join(DATASET_PATH, 'cifar-10-batches-py', 'data_batch_{}'.format(i))
        with open(data_file, 'rb') as fp:
            vals = pickle.load(fp, encoding='bytes')
            train_labels.extend(vals[b'labels'])
            train_filenames.extend(vals[b'filenames'])
            train_data.extend(vals[b'data'])

    test_labels = []
    test_filenames = []
    test_data = []
    with open(os.path.join(DATASET_PATH, 'cifar-10-batches-py', 'test_batch'), 'rb') as fp:
        vals = pickle.load(fp, encoding='bytes')
        test_labels.extend(vals[b'labels'])
        test_filenames.extend(vals[b'filenames'])
        test_data.extend(vals[b'data'])

    examples = []
    for lab, fn, dat in tqdm(zip(train_labels, train_filenames, train_data)):
        example = make_example(lab, PNG_PATH + 'train/' + fn.decode(), dat)
        example['fold'] = 'train'
        examples.append(example)

    for lab, fn, dat in tqdm(zip(test_labels, test_filenames, test_data)):
        example = make_example(lab, PNG_PATH + 'test/' + fn.decode(), dat)
        example['fold'] = 'test'
        examples.append(example)


    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]

    for idx, split in enumerate(splits):
        if not os.path.exists(DATASET_PATH + '/split' + str(idx)):
            os.mkdir(DATASET_PATH + '/split' + str(idx))
        unknown_classes = [cifar_class(i) for i in split]
        train_examples = {}
        train_examples['labels'] = []
        train_examples['data'] = []
        train_examples['filenames'] = []

        test_examples = {}
        test_examples['labels'] = []
        test_examples['data'] = []
        test_examples['filenames'] = []
        open_test_examples = {}
        open_test_examples['labels'] = []
        open_test_examples['data'] = []
        open_test_examples['filenames'] = []
        for i, e in enumerate(examples):
            if e['fold'] == 'train' and (e['class'] not in unknown_classes):
                train_examples['filenames'] += [e['filename'].encode()]
                train_examples['data'] += [e['data']]
                train_examples['labels'] += [e['label']]
            elif e['fold'] == 'test':
                if e['class'] not in unknown_classes:
                    test_examples['filenames'] += [e['filename'].encode()]
                    test_examples['data'] += [e['data']]
                    test_examples['labels'] += [e['label']]
                else:
                    open_test_examples['filenames'] += [e['filename'].encode()]
                    open_test_examples['data'] += [e['data']]
                    open_test_examples['labels'] += [e['label']]
        open_class_to_idx = {}
        open_idx_to_class = {}

        for fake_index, _idx in enumerate(split):
            open_idx_to_class[fake_index+6] = CIFAR_CLASSES[int(_idx)]
            open_class_to_idx[CIFAR_CLASSES[int(_idx)]] = fake_index+6

        class_to_idx = {}
        idx_to_class = {}

        meta_dict = {'num_cases_per_batch': 10000, 'num_vis': 3072}
        meta_dict['class_names'] = CIFAR_CLASSES
        fake_index = 0
        for _idx in range(10):
            if _idx not in split:
                idx_to_class[fake_index] = CIFAR_CLASSES[int(_idx)]
                class_to_idx[CIFAR_CLASSES[int(_idx)]] = fake_index
                fake_index += 1
        train_obj_ = '{}/split{}/train_obj.pkl'.format(DATASET_PATH, idx)
        test_obj_ = '{}/split{}/test_obj.pkl'.format(DATASET_PATH, idx)
        open_test_obj_ = '{}/split{}/open_test_obj.pkl'.format(DATASET_PATH, idx)
        idx_to_class_ = '{}/split{}/idx_to_class.pkl'.format(DATASET_PATH, idx)
        class_to_idx_ = '{}/split{}/class_to_idx.pkl'.format(DATASET_PATH, idx)

        meta_ = '{}/split{}/meta.pkl'.format(DATASET_PATH, idx)
        open_meta_ = '{}/split{}/open_meta.pkl'.format(DATASET_PATH, idx)

        open_class_to_idx_ = '{}/split{}/open_class_to_idx.pkl'.format(DATASET_PATH, idx)
        open_idx_to_class_ = '{}/split{}/open_idx_to_class.pkl'.format(DATASET_PATH, idx)

        save_image_dataset(train_examples, train_obj_)
        save_image_dataset(test_examples, test_obj_)
        save_image_dataset(open_test_examples, open_test_obj_)
        save_image_dataset(idx_to_class, idx_to_class_)
        save_image_dataset(class_to_idx, class_to_idx_)

        save_image_dataset(meta_dict, meta_)
        save_image_dataset(meta_dict, open_meta_)

        save_image_dataset(open_class_to_idx, open_class_to_idx_)
        save_image_dataset(open_idx_to_class, open_idx_to_class_)


    print("Finished writing datasets")


def make_example(label, filename, data):
    pixels = data.reshape(3, 32, 32)
    pixels = pixels.transpose(1, 2, 0)
    pixels = Image.fromarray(pixels)
    pixels.save(filename)
    class_name = cifar_class(label)
    return {
        'filename': filename,
        'data': pixels,
        'label': label,
        'class': class_name,
    }


def cifar_class(label_idx):
    return CIFAR_CLASSES[label_idx]


def is_animal(label):
    return label in ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']


def is_flying(label):
    return label in ['bird', 'airplane']


def is_pet(label):
    return label in ['dog', 'cat']


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


def listdir(path):
    filenames = os.listdir(os.path.expanduser(path))
    filenames = sorted(filenames)
    return [os.path.join(path, fn) for fn in filenames]


def download(file_path, file_name, url):
    filename = os.path.join(file_path, file_name)
    if not os.path.exists(filename):
        os.system('wget -P {} {}'.format(file_path, url))
    if filename.endswith('.tgz') or filename.endswith('.tar.gz'):
        os.system('tar xzvf {} -C {}'.format(filename, file_path))
    elif filename.endswith('.zip'):
        os.system('unzip *.zip')


def train_test_split(filename):
    # Training examples end with 0, test with 1, validation with 2
    return [line.strip().endswith('0') for line in open(filename)]


def save_image_dataset(examples, output_filename):

    with open(output_filename, 'wb') as fp:
        pickle.dump(examples, fp)
    print("Wrote {} items to {}".format(len(examples), output_filename))


if __name__ == '__main__':
    main()