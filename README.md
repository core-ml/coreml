# CoreML

`coreml` is an end-to-end machine learning framework aimed at supporting rapid prototyping. It is built on top of PyTorch by combining the several components of any ML pipeline, right from definining the dataset object, choosing how to sample each batch, preprocessing your inputs and labels, iterating on different network architectures, applying various weight initializations, running pretrained models, freezing certain layers, changing optimizers, adding learning rate schedulers, and detailed logging, into a simple `model.fit()` framework, similar to [`scikit-learn`](https://github.com/scikit-learn/scikit-learn). The codebase is very modular making it easily extensible for different tasks, modalities and training recipes, avoiding duplication wherever possible.   

## Features
- Support for end-to-end training using `PyTorch`.
- Makes every aspect of the training pipeline configurable.
- Provides the ability to define and change architectures right in the config file.
- Built-in support for experiment tracking using `Weights & Biases`.
- Supports tracking instance-level loss over epochs.
- Logs predictions and metrics over epochs to allow future analysis.
- Supports saving checkpoints and optimizing thresholds based on specific subsets.
- Defines several metrics like `PrecisionAtRecall`, `SpecificityAtSensitivity` and `ConfusionMatrix`.
- Logs several classification curves like `PR` curve, `Sensitivity-Specificity` curve, `ROC` curve.
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
$ docker pull adalmia/coreml:v1.0
```
OR

You can build the image using the `DockerFile`:
```bash
$ docker build -t adalmia/coreml:v1.0 .
```

The repository runs inside a Docker container. When creating the container, you need to mount the directory containing data to `/data` and directory where you want to store the ouptuts to `/output` on the container. Make the corresponding changes to `create_container.sh` to mount the respective directories by changing `/path/to/coreml`, `/path/to/data` and `/path/to/outputs` to the appropriate values.

Use the following command to launch a container:

```bash
$ bash create_container.sh
```

## Testing
We use `unittest` for all our tests. Simply run the following inside the Docker container:
```
$ python -m unittest discover tests
```
