"""Evaluation script

$ python evaluation/eval.py -v version
"""
import logging
import argparse
import os
from os.path import join, dirname, splitext
import multiprocessing as mp
import wandb
from coreml.config import Config
from coreml.data.dataloader import get_dataloader
from coreml.models import factory as model_factory
from coreml.utils.logger import set_logger, color


def evaluate(config, mode, use_wandb, ignore_cache):
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
    results = model.evaluate(dataloader, mode, use_wandb, ignore_cache)


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
    evaluate(config, args.mode, args.wandb, args.ignore_cache)


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
    args = parser.parse_args()
    main(args)
