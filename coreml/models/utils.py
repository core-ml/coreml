from coreml.data.utils import read_dataset_from_config


def get_subsets(subset_tracker_config):
    mode_subsets = dict()
    for subset_config in subset_tracker_config:
        # each subset has its own data config with a corresponding
        # `mode` and we keep a dictionary of subset `mode` and the
        # corresponding IDs
        subset_info = read_dataset_from_config(subset_config)

        # converting to set as comparison becomes faster than a list
        mode_subsets[subset_config['mode']] = set(
            subset_info['file'])

    return mode_subsets
