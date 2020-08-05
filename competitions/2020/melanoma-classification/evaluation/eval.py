"""Evaluation script

$ python evaluation/eval.py -v version
"""
import logging
import warnings
import argparse
import os
from os.path import join, dirname, splitext, basename
import multiprocessing as mp
import pandas as pd
import wandb
import torch
from coreml.config import Config
from coreml.data.dataloader import get_dataloader
from coreml.models import factory as model_factory
from coreml.utils.logger import set_logger, color
warnings.simplefilter('ignore')


def evaluate(config, mode, use_wandb, ignore_cache, n_tta):
    """Run the actual evaluation

    :param config: config for the model to evaluate
    :type config: Config
    :param mode: data mode to evaluate on
    :type mode: str
    :param use_wandb: whether to log values to wandb
    :type use_wandb: bool
    :param ignore_cache: whether to ignore cached predictions
    :type ignore_cache: bool
    """
    model = model_factory.create(config.model['name'], **{'config': config})
    logging.info(color(f'Evaluating on mode: {mode}'))
    dataloader, _ = get_dataloader(
        config.data, mode,
        config.model['batch_size'],
        num_workers=config.num_workers,
        shuffle=False,
        drop_last=False)

    # set to eval mode
    model.network.eval()

    all_predictions = []
    for run_index in range(n_tta):
        logging.info(f'TTA run #{run_index + 1}')
        results = model.evaluate(
            dataloader, mode, use_wandb,
            ignore_cache, data_only=True, log_summary=False)

        logging.info(f'AUC = {results["auc-roc"]}')

        # logits
        predictions = results['predictions']

        # convert to softmax
        predictions = torch.sigmoid(predictions)

        # add to list of all predictions across each TTA run
        all_predictions.append(predictions)

    all_predictions = torch.stack(all_predictions, -1)

    # take the mean across several TTA runs
    predictions = all_predictions.mean(-1)

    # get the file names
    names = [splitext(basename(item.path))[0] for item in results['items']]

    # convert to data frame
    data_frame = pd.DataFrame({
        'image_name': names, 'target': predictions.tolist()
    })

    # save the results
    save_path = join(config.log_dir, 'evaluation', f'{mode}.csv')
    os.makedirs(dirname(save_path), exist_ok=True)
    logging.info(color(f'Saving results to {save_path}'))
    data_frame.to_csv(save_path, index=False)


def main(args):
    version = args.version
    config = Config(version)
    version = splitext(version)[0]

    set_logger(join(config.log_dir, 'eval.log'))
    logging.info(args)

    # add checkpoint loading values
    load_epoch = args.epoch
    load_best = args.best
    config.model['load']['version'] = version
    config.model['load']['epoch'] = load_epoch
    config.model['load']['load_best'] = load_best

    # ensures that the epoch_counter attribute is set to the
    # epoch number being loaded
    config.model['load']['resume_epoch'] = True

    if args.wandb:
        # set up wandb
        os.environ['WANDB_ENTITY'] = args.entity
        os.environ['WANDB_PROJECT'] = args.project
        os.environ['WANDB_DIR'] = dirname(config.checkpoint_dir)

        run_name = '_'.join(['evaluation', version.replace('/', '_')])
        wandb.init(name=run_name, dir=dirname(config.checkpoint_dir),
                   notes=config.description)
        wandb.config.update(config.__dict__)

    config.num_workers = args.num_workers
    evaluate(config, args.mode, args.wandb, args.ignore_cache, args.n_tta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates a model")
    parser.add_argument('-v', '--version', required=True, type=str,
                        help='path to the experiment config file')
    parser.add_argument('-n', '--num_workers', default=mp.cpu_count(),
                        type=int, help='number of CPU workers to use')
    parser.add_argument('-m', '--mode', type=str, default='test',
                        help='specifies the split of data to evaluate on')
    parser.add_argument('--epoch', type=int, default=-1,
                        help='specifies the checkpoint epoch to load')
    parser.add_argument('-b', '--best', action='store_true',
                        help='whether to load the best saved checkpoint')
    parser.add_argument('-i', '--ignore-cache', action='store_true',
                        help='whether to ignore cache')
    parser.add_argument('--wandb', action='store_false',
                        help='whether to ignore using wandb')
    parser.add_argument('-e', '--entity', type=str,
                        help='wandb user/org name')
    parser.add_argument('-p', '--project', type=str,
                        help='wandb project name')
    parser.add_argument('-o', '--output', type=str,
                        help='wandb project name')
    parser.add_argument('--n-tta', type=int, default=1,
                        help='number of times to run TTA')
    args = parser.parse_args()
    main(args)
