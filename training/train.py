"""Script to run training"""
import warnings
import logging
import argparse
import os
from os.path import join, dirname
import multiprocessing as mp
import wandb
import pytorch_lightning as pl
from coreml.config import Config
from coreml.trainer import Trainer

warnings.simplefilter('ignore')

# TODO: notes in wandb
# TODO: allow_val_change in wandb
# TODO: on fit end log best_model_path and best_model_score to summary wandb


# def train(config, debug, overfit_batch, use_wandb):
    
    # model.fit(debug=debug, overfit_batch=overfit_batch, use_wandb=use_wandb)


def main(args):
    pl.seed_everything(args.seed)
    config = Config(args.version)

    config.trainer['num_workers'] = args.num_workers
    trainer_args = vars(args)

    # setup wandb
    if args.wandb:
        config.logger.update({
            'name': args.version.replace('/', '_'),
            'save_dir': dirname(config.checkpoint_dir),
            'id': args.id,
            'offline': args.wandb
        })
        logger = pl.loggers.WandbLogger(**config.logger)
        trainer_args['logger'] = logger

    # remove redundant keys from args
    keys_to_remove = ['version', 'num_workers', 'id', 'wandb', 'seed']
    for key in keys_to_remove:
        trainer_args.pop(key, None)

    # override default args with args set in config
    trainer_args.update(config.trainer['params'])

    # define trainer object
    trainer = Trainer(config, **trainer_args)

    # train the model
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains a model")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-v', '--version', required=True, type=str,
                        help='path to the experiment config file')
    parser.add_argument('-n', '--num_workers', default=mp.cpu_count(),
                        type=int, help='number of CPU workers to use')
    parser.add_argument('--id', type=str, default=None,
                        help='experiment ID in wandb')
    parser.add_argument('--wandb', action='store_false',
                        help='whether to ignore using wandb')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for the experiment')
    args = parser.parse_args()
    main(args)
