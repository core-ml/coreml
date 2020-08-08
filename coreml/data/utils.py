from typing import Tuple
from os.path import join
import sys
from coreml.utils.logger import color
from coreml.utils.io import read_yml
from coreml.utils.typing import DatasetConfigDict


def read_dataset_from_config(
        data_root: str, dataset_config: DatasetConfigDict) -> dict:
    """
    Loads and returns the dataset version file corresponding to the
    dataset config.

    :param data_root: directory where data versions reside
    :type dataset_config: DatasetConfigDict
    :param dataset_config: dict containing `(name, version, mode)`
        corresponding to a dataset. Here, `name` stands for the name of the
        dataset under the `/data` directory, `version` stands for the version
        of the dataset (stored in `/data/name/processed/versions/`) and `mode`
        stands for the split to be loaded (train/val/test).
    :type dataset_config: DatasetConfigDict
    :returns: dict of values stored in the version file
    """
    version_fpath = join(
        data_root, dataset_config['name'],
        'processed/versions', dataset_config['version'] + '.yml')

    print(color("=> Loading dataset version file: [{}, {}, {}]".format(
        dataset_config['name'], dataset_config['version'],
        dataset_config['mode'])))

    version_file = read_yml(version_fpath)

    return version_file[dataset_config['mode']]
