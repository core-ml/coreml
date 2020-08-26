# CoreML

`coreml` is an end-to-end machine learning framework aimed at supporting rapid prototyping. It is built on top of `PyTorchLightning` by combining the several components of any ML pipeline, right from definining the dataset object, choosing how to sample each batch, preprocessing your inputs and labels, iterating on different network architectures, applying various weight initializations, running pretrained models, freezing certain layers, changing optimizers, adding learning rate schedulers, and detailed logging, into a simple `model.fit()` framework, similar to [`scikit-learn`](https://github.com/scikit-learn/scikit-learn). The codebase is very modular making it easily extensible for different tasks, modalities and training recipes, avoiding duplication wherever possible. The contents of the README are as follows:

- [Features](#features)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [How-Tos](#how-tos)
  - [Optimization](#optimization)
  - [Network architectures](#network-architectures)
  - [Datasets](#datasets)
  - [Preprocessing](#preprocessing)
- [Testing](#testing)
- [TODOs](#todos)
- [Authors](#authors)

## Features
- Support for end-to-end training using `PyTorchLightning` with custom training and validation loops.
- Makes every aspect of the training pipeline configurable.
- Data preprocessing on GPU using `kornia`.
- Provides the ability to define and change architectures right in the config file.
- Built-in support for experiment tracking using `Weights & Biases`.
- Enables replicability through `Docker` containers.
- Defines several new metrics like `PrecisionAtRecall`, `SpecificityAtSensitivity` and `ConfusionMatrix`.
- Logs several classification curves like `PR` curve, `Sensitivity-Specificity` curve and `ROC` curve.
- Explicitly requires data versioning.
- Supports adding new datasets adhering to a required format.
- Contains unit tests wherever applicable.

## Setup

Clone the project:

```bash
$ git clone https://github.com/dalmia/coreml.git
```

### Weights & Biases
We use [`wandb`](http://wandb.com/) for experiment tracking. You'll need to have that set up:
- Install wandb

```bash
$ pip install wandb
```

2. Login to wandb:

```bash
$ wandb login
```

You will be redirected to a link that will show you your `WANDB_API_KEY` .

3. Set the `WANDB_API_KEY` by adding this to your `~/.bashrc` file:
```bash
export WANDB_API_KEY=YOUR_API_KEY
```

4. Run `source ~/.bashrc`.
5. During training, you'll have an option to turn off `wandb` as well.


### Docker
We use Docker containers to ensure replicability of experiments. You can either fetch the Docker image from DockerHub using the following line:
```bash
$ docker pull adalmia/coreml:v1.0-lightning
```
OR

You can build the image using the `DockerFile`:
```bash
$ docker build -t adalmia/coreml:v1.0-lightning .
```

The repository runs inside a Docker container. When creating the container, you need to mount the directory containing data to `/data` and directory where you want to store the ouptuts to `/output` on the container. Make the corresponding changes to `create_container.sh` to mount the respective directories by changing `/path/to/coreml`, `/path/to/data` and `/path/to/outputs` to the appropriate values.

Use the following command to launch a container:

```bash
$ bash create_container.sh
```

## Quickstart
### CIFAR10
- Download and prepare the data
```
$ python tasks/data/classification/CIFAR.py
```

This will create a folder named `CIFAR10` under the directory mounted at `/data` on your container along with default data versions and the required directory structure.

- Since the codebase only supports binary classification for now, the step above also creates a `binary.yml` data version which converts the 10-class problem into a binary classification problem.

- Run training using the default config:
```
$ python training/train.py -v configs/defaults/binary-cifar-classification.yml
```

The `training/train.py` adds a few more flags on top of the ones defined by [`Trainer`](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api) from `PyTorchLightning` You can view all the flags that can be passed using `-h`:
```
root@ip:/workspace/coreml# python training/train.py -h

usage: train.py [-h] [--logger [LOGGER]]
                [--checkpoint_callback [CHECKPOINT_CALLBACK]]
                [--early_stop_callback [EARLY_STOP_CALLBACK]]
                [--default_root_dir DEFAULT_ROOT_DIR]
                [--gradient_clip_val GRADIENT_CLIP_VAL]
                [--process_position PROCESS_POSITION] [--num_nodes NUM_NODES]
                [--num_processes NUM_PROCESSES] [--gpus GPUS]
                [--auto_select_gpus [AUTO_SELECT_GPUS]]
                [--tpu_cores TPU_CORES] [--log_gpu_memory LOG_GPU_MEMORY]
                [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE]
                [--overfit_batches OVERFIT_BATCHES]
                [--track_grad_norm TRACK_GRAD_NORM]
                [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH]
                [--fast_dev_run [FAST_DEV_RUN]]
                [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
                [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS]
                [--max_steps MAX_STEPS] [--min_steps MIN_STEPS]
                [--limit_train_batches LIMIT_TRAIN_BATCHES]
                [--limit_val_batches LIMIT_VAL_BATCHES]
                [--limit_test_batches LIMIT_TEST_BATCHES]
                [--val_check_interval VAL_CHECK_INTERVAL]
                [--log_save_interval LOG_SAVE_INTERVAL]
                [--row_log_interval ROW_LOG_INTERVAL]
                [--distributed_backend DISTRIBUTED_BACKEND]
                [--sync_batchnorm [SYNC_BATCHNORM]] [--precision PRECISION]
                [--weights_summary WEIGHTS_SUMMARY]
                [--weights_save_path WEIGHTS_SAVE_PATH]
                [--num_sanity_val_steps NUM_SANITY_VAL_STEPS]
                [--truncated_bptt_steps TRUNCATED_BPTT_STEPS]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--profiler [PROFILER]] [--benchmark [BENCHMARK]]
                [--deterministic [DETERMINISTIC]]
                [--reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]]
                [--auto_lr_find [AUTO_LR_FIND]]
                [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]]
                [--terminate_on_nan [TERMINATE_ON_NAN]]
                [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]]
                [--prepare_data_per_node [PREPARE_DATA_PER_NODE]]
                [--amp_backend AMP_BACKEND] [--amp_level AMP_LEVEL]
                [--val_percent_check VAL_PERCENT_CHECK]
                [--test_percent_check TEST_PERCENT_CHECK]
                [--train_percent_check TRAIN_PERCENT_CHECK]
                [--overfit_pct OVERFIT_PCT] -v VERSION [-n NUM_WORKERS]
                [--id ID] [--wandb] [--seed SEED]

Trains a model

optional arguments:
  -h, --help            show this help message and exit
  --logger [LOGGER]     autogenerated by pl.Trainer
  --checkpoint_callback [CHECKPOINT_CALLBACK]
                        autogenerated by pl.Trainer
  --early_stop_callback [EARLY_STOP_CALLBACK]
                        autogenerated by pl.Trainer
  --default_root_dir DEFAULT_ROOT_DIR
                        autogenerated by pl.Trainer
  --gradient_clip_val GRADIENT_CLIP_VAL
                        autogenerated by pl.Trainer
  --process_position PROCESS_POSITION
                        autogenerated by pl.Trainer
  --num_nodes NUM_NODES
                        autogenerated by pl.Trainer
  --num_processes NUM_PROCESSES
                        autogenerated by pl.Trainer
  --gpus GPUS           autogenerated by pl.Trainer
  --auto_select_gpus [AUTO_SELECT_GPUS]
                        autogenerated by pl.Trainer
  --tpu_cores TPU_CORES
                        autogenerated by pl.Trainer
  --log_gpu_memory LOG_GPU_MEMORY
                        autogenerated by pl.Trainer
  --progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE
                        autogenerated by pl.Trainer
  --overfit_batches OVERFIT_BATCHES
                        autogenerated by pl.Trainer
  --track_grad_norm TRACK_GRAD_NORM
                        autogenerated by pl.Trainer
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        autogenerated by pl.Trainer
  --fast_dev_run [FAST_DEV_RUN]
                        autogenerated by pl.Trainer
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        autogenerated by pl.Trainer
  --max_epochs MAX_EPOCHS
                        autogenerated by pl.Trainer
  --min_epochs MIN_EPOCHS
                        autogenerated by pl.Trainer
  --max_steps MAX_STEPS
                        autogenerated by pl.Trainer
  --min_steps MIN_STEPS
                        autogenerated by pl.Trainer
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        autogenerated by pl.Trainer
  --limit_val_batches LIMIT_VAL_BATCHES
                        autogenerated by pl.Trainer
  --limit_test_batches LIMIT_TEST_BATCHES
                        autogenerated by pl.Trainer
  --val_check_interval VAL_CHECK_INTERVAL
                        autogenerated by pl.Trainer
  --log_save_interval LOG_SAVE_INTERVAL
                        autogenerated by pl.Trainer
  --row_log_interval ROW_LOG_INTERVAL
                        autogenerated by pl.Trainer
  --distributed_backend DISTRIBUTED_BACKEND
                        autogenerated by pl.Trainer
  --sync_batchnorm [SYNC_BATCHNORM]
                        autogenerated by pl.Trainer
  --precision PRECISION
                        autogenerated by pl.Trainer
  --weights_summary WEIGHTS_SUMMARY
                        autogenerated by pl.Trainer
  --weights_save_path WEIGHTS_SAVE_PATH
                        autogenerated by pl.Trainer
  --num_sanity_val_steps NUM_SANITY_VAL_STEPS
                        autogenerated by pl.Trainer
  --truncated_bptt_steps TRUNCATED_BPTT_STEPS
                        autogenerated by pl.Trainer
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        autogenerated by pl.Trainer
  --profiler [PROFILER]
                        autogenerated by pl.Trainer
  --benchmark [BENCHMARK]
                        autogenerated by pl.Trainer
  --deterministic [DETERMINISTIC]
                        autogenerated by pl.Trainer
  --reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]
                        autogenerated by pl.Trainer
  --auto_lr_find [AUTO_LR_FIND]
                        autogenerated by pl.Trainer
  --replace_sampler_ddp [REPLACE_SAMPLER_DDP]
                        autogenerated by pl.Trainer
  --terminate_on_nan [TERMINATE_ON_NAN]
                        autogenerated by pl.Trainer
  --auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]
                        autogenerated by pl.Trainer
  --prepare_data_per_node [PREPARE_DATA_PER_NODE]
                        autogenerated by pl.Trainer
  --amp_backend AMP_BACKEND
                        autogenerated by pl.Trainer
  --amp_level AMP_LEVEL
                        autogenerated by pl.Trainer
  --val_percent_check VAL_PERCENT_CHECK
                        autogenerated by pl.Trainer
  --test_percent_check TEST_PERCENT_CHECK
                        autogenerated by pl.Trainer
  --train_percent_check TRAIN_PERCENT_CHECK
                        autogenerated by pl.Trainer
  --overfit_pct OVERFIT_PCT
                        autogenerated by pl.Trainer
  -v VERSION, --version VERSION
                        path to the experiment config file
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        number of CPU workers to use
  --id ID               experiment ID in wandb
  --wandb               whether to use wandb
  --seed SEED           seed for the experiment
```

## How-Tos
This section demonstrates the power of everything being parameterized by a config. Refer to `configs/defaults/binary-cifar.yml`
as the base config file on top of which we demonstrate the individual components.

### Optimization

```yaml
optimizer:
    name: SGD
    args:
      lr: 0.001
      momentum: 0.9
      nesterov: true
    scheduler:
      name: StepLR
      init_params:
          gamma: 0.9
          step_size: 1
      opt_params:
        interval: epoch

loss:
    train:
      name: binary-cross-entropy
      params:
          reduction: none
    val:
      name: binary-cross-entropy
      params:
          reduction: none
    test:
      name: binary-cross-entropy
      params:
          reduction: none
```
The above example shows how to set various hyperparameters like the batch size, number of epochs, optimizer, learning rate scheduler and the loss function. The interesting aspect is how the optimizer, learning rate scheduler and the loss function is directly defined in the config file. This is possible because of the [Factory Design Pattern](https://www.tutorialspoint.com/design_pattern/factory_pattern.htm) used throughout the codebase. For `optimizer`, we currently support:
- `SGD`
- `Adam`
- `AdamW`

However, other optimization functions can be simply added by registering their corresponding builders in `coreml/modules/optimization.py`. For each optimizer, `args` contains any parameters required by their corresponding `PyTorch` definition.

Similarly, we support multiple learning rate schedulers defined in `PyTorch` along parameterizing whether the scheduler's step should take place after each batch or after each epoch. This is controlled by the key `interval`, which can be one of `['epoch', 'step']`. The key `monitor` can be set to decide what parameter should be monitored for the scheduler's step. In the above example, the validation `loss` is being monitored. Currently, we suport the following schedulers:
- `ReduceLROnPlateau`
- `StepLR`
- `CyclicLR`
- `OneCycleLR`
- `MultiStepLR`

We also parameterize the loss function to be used and allow for different loss functions for training and validation. The need for making them different could arise in various situations. One such example is applying label smoothing during training but not during validation.

### Network architectures
The network architecture can be completely defined in the config itself:
```yaml
network:
    name: neural_net
    params:
        config:
        - name: Conv2d
          params:
            in_channels: 3
            out_channels: 64
            kernel_size: 3
        - name: BatchNorm2d
          params:
            num_features: 64
        - name: ReLU
          params: {}
        - name: Conv2d
          params:
            in_channels: 64
            out_channels: 64
            kernel_size: 3
        - name: BatchNorm2d
          params:
            num_features: 64
        - name: ReLU
          params: {}
        - name: AdaptiveAvgPool2d
          params:
            output_size:
            - 1
            - 1
        - name: Flatten
          params: {}
        - name: Linear
          params:
            in_features: 64
            out_features: 64
        - name: ReLU
          params: {}
        - name: Linear
          params:
            in_features: 64
            out_features: 1
```
The `config` key takes as input a list of dictionaries, with each dictionary specifying a layer or a backbone network. Yes, if you want to use a pretrained ResNet, you can simply plug it in as a backbone layer:

```yaml
network:
    name: neural_net
    params:
        config:
        - name: resnet50
          params:
            pretrained: true
            in_channels: 3
        - name: AdaptiveAvgPool2d
          params:
            output_size:
            - 1
            - 1
        - name: Flatten
          params: {}
        - name: Linear
          params:
            in_features: 2048
            out_features: 1
```
Currently, we support a lot of backbones:
- `Resnet` variations: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`
- `VGGNet` variations: `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19_bn`, `vgg19`
-  `EfficientNet` variations: `efficientnet-b0`, `efficientnet-b4`, `efficientnet-b7`

The implementations for the `Resnet` and `VGGNet` based backbones have been taken from `torchvision.models` and those based on `EfficientNet` are supported by [this](https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch) PyTorch implementation.

Support for other backbones can be similarly added to `coreml/modules/backbone`.

### Datasets
The datasets to use can be specified as follows:
```yaml
dataset:
    name: classification_dataset
    config:
      - name: CIFAR10
        version: binary
    params: {}
```
Here, `name` is used to decide which `torch.utils.data.Dataset` object to use, as defined in `coreml/data/__init__.py`. `config` takes a list of dataset configs. Each dataset config is a dictionary with `name` as the name of the dataset folder under `/data` and `version` being the specific dataset version to use, present in `/data/{name}/processed/versions`. `params` contains additional arguments that can be passed to the dataset object, as defined in `coreml/data`. When values are passed to `params`, they are specified for `train`/`val`/`test` separately:
```yaml
dataset:
    name: classification_dataset
    config:
      - name: CIFAR10
        version
    params:
      train:
        fraction: 0.1
      val:
        fraction: 0.5
```

### Preprocessing

#### Input transform
One the input is loaded, the pipeline for processing the input before it is fed into a batch, can be specified in the config as:
```yaml
signal_transform:
    train:
    - name: Permute
      params:
        order:
          - 2
          - 0
          - 1
    val:
    - name: Permute
      params:
        order:
          - 2
          - 0
          - 1
```
The pipeline for each split (`train`/`val`) is specified separately. For each split, a list of dictonaries are given. Each dictionary
represents one transform, as defined in `coreml/data/transforms.py`, using the `name` of the transform and the arguments for that transform.

#### Annotation transform
The raw annotations might have to be transformed before processing as well. This is specified in the config as:
```yaml
target_transform:
    name: classification
    params:
      classes:
        - 0
        - 1
```
The specific target transform is selected from `annotation_factory` inside `coreml/data/transforms.py`

## Testing
We use `unittest` for all our tests. Simply run the following inside the Docker container:
```
$ python -m unittest discover tests
```

## TODOs
- Tracking inputs on W&B (currently `torch.cat` kills the process when the inputs are large).
- Add augmentations
- Label Smoothing for BCE
- Incorporate models from `timm` (pytorch-image-models)
- Training with TPUs
- Training with multiple GPUs
- Native TTA
- Stochastic Weight Averaging
- Demos for various things
- Support for audio classification
- Classification analyzer (similar to fast.ai)
- Discriminative fine-tuning (see [this](https://course.fast.ai/videos/?lesson=1))
- Add benchmarks for multiple datasets
- Add documentation for using new datasets and configuring different parts of the pipeline

## Authors
- [Aman Dalmia](https://github.com/dalmia)
- [Piyush Bagad](https://github.com/bpiyush)
