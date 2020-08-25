"""Evaluation script

$ python evaluation/eval.py -v version
"""
import logging
import warnings
import argparse
from os.path import join, dirname, splitext, basename
import multiprocessing as mp
import pytorch_lightning as pl
from coreml.config import Config
from coreml.trainer import Trainer
from coreml.utils.logger import color
from coreml.utils.checkpoint import get_last_saved_checkpoint_path
warnings.simplefilter('ignore')


def main(args):
    config = Config(args.version)
    version = splitext(args.version)[0]

    if args.bs is not None:
        config.trainer['batch_size'] = args.bs

    config.trainer['num_workers'] = args.num_workers
    trainer_args = vars(args)

    # disable logger by default
    trainer_args['logger'] = None

    # setup wandb
    if args.wandb:
        config.logger.update({
            'name': '_'.join(['evaluation', version.replace('/', '_')]),
            'save_dir': dirname(config.checkpoint_dir),
        })
        logger = pl.loggers.WandbLogger(**config.logger)
        trainer_args['logger'] = logger

    print(color(f'Evaluating on mode: {args.mode}'))
    eval_mode = args.mode
    ckpt_path = args.ckpt_path

    # reset sampler to default
    config.data['sampler'].update({
        eval_mode: {
            'name': 'default'
        }
    })

    # remove redundant keys from args
    keys_to_remove = [
        'version', 'num_workers', 'id', 'mode', 'wandb',
        'ckpt_path', 'bs', 'ignore_cache'
    ]
    for key in keys_to_remove:
        trainer_args.pop(key, None)

    if ckpt_path == -1:
        # restore trainer to the state at the end of training
        # ISSUE: ideally, it should resume optimizer state and callback states
        # as well - that does not happen - lightning issue
        ckpt_path = get_last_saved_checkpoint_path(config.checkpoint_dir)

    trainer_args['resume_from_checkpoint'] = ckpt_path

    # log which checkpoint is going to be used
    print(color(f'Using checkpoint as: {ckpt_path}'))

    # define trainer object
    trainer = Trainer(config, **trainer_args)

    # run evaluation
    trainer.evaluate(eval_mode, ckpt_path=ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates a model")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-v', '--version', required=True, type=str,
                        help='path to the experiment config file')
    parser.add_argument('-n', '--num_workers', default=mp.cpu_count(),
                        type=int, help='number of CPU workers to use')
    parser.add_argument('-m', '--mode', type=str, default='test',
                        help='specifies the split of data to evaluate on')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='specifies the checkpoint path to load')
    parser.add_argument('--bs', default=None, type=int,
                        help='batch size to use')
    parser.add_argument('-i', '--ignore-cache', action='store_true',
                        help='whether to ignore cache')
    parser.add_argument('--wandb', action='store_false',
                        help='whether to ignore using wandb')
    args = parser.parse_args()
    main(args)
