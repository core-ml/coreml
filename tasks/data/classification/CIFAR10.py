import os
from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from torchvision.datasets import CIFAR10
from coreml.utils.io import save_yml

# load train split
train = CIFAR10('/data/CIFAR10/raw',  download=True)

# load test split
test = CIFAR10('/data/CIFAR10/raw', train=False, download=True)

processed_dir = '/data/CIFAR10/processed'
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
    cv2.imwrite(image_path, image[:, :, ::-1])


splits = ['train'] * len(train.data) + ['test'] * len(test.data)
labels = [{'classification': [all_targets[index]]} for index in range(len(all_targets))]

annotation = pd.DataFrame({'path': image_paths, 'label': labels, 'split': splits})

# save annotation data
annotation.to_csv(annotation_path, index=False)

# create version data
version = {}

# train split
version['train'] = {
    'file': image_paths[:len(train.data)],
    'label': all_targets[:len(train.data)]
}

# test split
version['test'] = {
    'file': image_paths[len(train.data):],
    'label': all_targets[len(train.data):]
}

# check shapes
assert len(version['train']['file']) == 50000
assert len(version['test']['file']) == 10000

# save the version file
print(f'Saving version file to {version_path}')
save_yml(version_path, version)
