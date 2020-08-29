import os
from os.path import join, exists
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from torchvision.datasets import CIFAR100
from sklearn.model_selection import train_test_split
from coreml.utils.io import save_yml


def download_and_process(data_dir):
    # load train split
    train = CIFAR100(join(data_dir, 'CIFAR100/raw'), download=True)

    # load test split
    test = CIFAR100(join(data_dir, 'CIFAR100/raw'), train=False, download=True)

    processed_dir = join(data_dir, 'CIFAR100/processed')
    os.makedirs(processed_dir, exist_ok=True)

    print(f'Train data shape: {train.data.shape}')
    print(f'Test data shape: {test.data.shape}')

    # concatenate all data
    all_images = np.append(train.data, test.data, axis=0)
    all_targets = np.append(train.targets, test.targets, axis=0)

    print(f'All data shape: {all_images.shape}')

    # define directories and paths
    image_dir = join(processed_dir, 'images')
    annotation_path = join(processed_dir, 'annotation.csv')
    version_dir = join(processed_dir, 'versions')
    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    version_path = join(version_dir, 'default.yml')

    # save images
    image_paths = []
    for index in tqdm(range(len(all_images)), desc='Saving images'):
        image = all_images[index]
        image_path = join(image_dir, f'{index}.png')
        image_paths.append(image_path)
        if not exists(image_path):
            cv2.imwrite(image_path, image[:, :, ::-1])

    splits = ['train'] * len(train.data) + ['test'] * len(test.data)
    labels = [{
        'classification': [all_targets[index].tolist()]
        } for index in range(len(all_targets))]

    annotation = pd.DataFrame({
        'path': image_paths, 'label': labels, 'split': splits})

    # save annotation data
    annotation.to_csv(annotation_path, index=False)

    # train-val split
    train_indices, val_indices = train_test_split(
        np.arange(len(train.data)), test_size=0.2, random_state=0)
    train_image_paths = [image_path for index, image_path in enumerate(
        image_paths) if index in train_indices]
    val_image_paths = [image_path for index, image_path in enumerate(
        image_paths) if index in val_indices]
    assert len(train_image_paths) == 40000
    assert len(val_image_paths) == 10000

    train_labels = [label for index, label in enumerate(
        labels) if index in train_indices]
    val_labels = [label for index, label in enumerate(
        labels) if index in val_indices]
    assert len(train_labels) == 40000
    assert len(val_labels) == 10000

    # create version data
    version = {}

    # train split
    version['train'] = {
        'file': train_image_paths,
        'label': train_labels
    }

    # val split
    version['val'] = {
        'file': val_image_paths,
        'label': val_labels
    }

    # train + val split
    version['train-val'] = {
        'file': train_image_paths + val_image_paths,
        'label': train_labels + val_labels
    }

    # test split
    version['test'] = {
        'file': image_paths[len(train.data):],
        'label': labels[len(train.data):]
    }

    # check shapes
    assert len(version['train']['file']) == 40000
    assert len(version['val']['file']) == 10000
    assert len(version['train-val']['file']) == 50000
    assert len(version['test']['file']) == 10000

    # check label types
    assert isinstance(version['train']['label'], list)
    assert isinstance(version['train']['label'][0], dict)
    assert isinstance(version['val']['label'], list)
    assert isinstance(version['val']['label'][0], dict)
    assert isinstance(version['test']['label'], list)
    assert isinstance(version['test']['label'][0], dict)

    # save the version file
    print(f'Saving version file to {version_path}')
    save_yml(version_path, version)

    # ------- create the binary CIFAR100 version ------- #
    binary_classes = [0, 1]
    print('Get binary indices')
    binary_indices = np.where(
        (all_targets == binary_classes[0]) | (all_targets == binary_classes[1])
    )[0]
    train_binary_indices = [index for index in binary_indices if index < 50000]
    test_binary_indices = [index for index in binary_indices if index >= 50000]

    print('Get splits')
    train_indices, val_indices = train_test_split(
        train_binary_indices, test_size=0.2)

    # convert to set for faster search
    train_indices = set(train_indices)
    val_indices = set(val_indices)
    test_binary_indices = set(test_binary_indices)

    train_image_paths = [image_path for index, image_path in enumerate(
        image_paths) if index in train_indices]
    val_image_paths = [image_path for index, image_path in enumerate(
        image_paths) if index in val_indices]
    test_image_paths = [image_path for index, image_path in enumerate(
        image_paths) if index in test_binary_indices]

    train_labels = [label for index, label in enumerate(
        labels) if index in train_indices]
    val_labels = [label for index, label in enumerate(
        labels) if index in val_indices]
    test_labels = [label for index, label in enumerate(
        labels) if index in test_binary_indices]

    for _labels in [train_labels, val_labels, test_labels]:
        unique_labels = set()
        for _label in _labels:
            unique_labels.add(_label['classification'][0])

        assert unique_labels == {0, 1}

    # create version data
    version = {}

    # train split
    version['train'] = {
        'file': train_image_paths,
        'label': train_labels
    }

    # val split
    version['val'] = {
        'file': val_image_paths,
        'label': val_labels
    }

    # train + val split
    version['train-val'] = {
        'file': train_image_paths + val_image_paths,
        'label': train_labels + val_labels
    }

    # test split
    version['test'] = {
        'file': test_image_paths,
        'label': test_labels
    }

    # check shapes
    assert len(train_labels) + len(val_labels) + len(
        test_labels) == len(binary_indices)

    # check label types
    assert isinstance(version['train']['label'], list)
    assert isinstance(version['train']['label'][0], dict)
    assert isinstance(version['val']['label'], list)
    assert isinstance(version['val']['label'][0], dict)
    assert isinstance(version['test']['label'], list)
    assert isinstance(version['test']['label'][0], dict)

    # save the binary version file
    version_path = join(version_dir, 'binary.yml')
    print(f'Saving version file to {version_path}')
    save_yml(version_path, version)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download and prepare CIFAR100")
    parser.add_argument('-d', '--data', default='/data', type=str,
                        help='path where the CIFAR100 dataset will be stored')
    args = parser.parse_args()
    download_and_process(args.data)
