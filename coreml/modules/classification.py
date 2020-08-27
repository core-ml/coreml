"""Defines the class for feed-forward LightningModule."""
from typing import Dict, Tuple, Any, List
import torch
import numpy as np
from pytorch_lightning.metrics.classification import Accuracy, Precision,\
    Recall, AUROC, PrecisionRecall, ROC
import matplotlib.pyplot as plt
import wandb
from coreml.modules.nn import NeuralNetworkModule
from coreml.utils.loss import loss_factory
from coreml.utils.metrics import metric_factory, ConfusionMatrix
from coreml.utils.viz import plot_classification_metric_curve
from coreml.utils.wandb import get_confusion_matrix
from coreml.utils.logger import color


class BinaryClassificationModule(NeuralNetworkModule):
    """LightningModule for binary classification"""
    def calculate_instance_loss(
            self, predictions: torch.Tensor, targets: torch.Tensor,
            mode: str) -> dict:
        """Calculate loss per instance in a batch

        :param predictions: Predictions (Predicted)
        :type predictions: torch.Tensor
        :param targets: Targets (Ground Truth)
        :type targets: torch.Tensor
        :param mode: train/val/test mode
        :type mode: str

        :return: dict of losses with list of loss values per instance
        """
        loss_config = self.config['loss'][mode]
        criterion = loss_factory.create(
            loss_config['name'], **loss_config['params'])

        # correct data type to handle mismatch between
        # CrossEntropyLoss and BCEWithLogitsLoss
        if loss_config['name'] == 'cross-entropy':
            targets = targets.long()

        loss = criterion(predictions, targets)
        return {'loss': loss}

    def compute_epoch_metrics(
            self, predictions: torch.Tensor, targets: torch.Tensor,
            threshold: float = None, recall: float = 0.9,
            as_logits: bool = True, classes: List[str] = None) -> dict:
        """Computes metrics for the epoch

        :param predictions: epoch predictions
        :type predictions: torch.Tensor
        :param targets: ground truths for the epoch
        :type targets: torch.Tensor
        :param threshold: confidence threshold to be used for binary
            classification; if None, the optimal threshold is found.
        :type threshold: float, defaults to None
        :param recall: minimum recall to choose the optimal threshold
        :type recall: float, defaults to 0.9
        :param as_logits: whether the predictions are logits; if
            as_logits=True, the values are converted into sigmoid scores
            before further processing.
        :type as_logits: bool, defaults to True
        :param classes: list of classes in the target
        :type classes: List[str], defaults to None

        :return: dictionary of metrics
        """
        if as_logits:
            # convert to sigmoid scores from logits
            predictions = torch.sigmoid(predictions)

        targets = targets.cpu()
        predict_proba = predictions.detach().cpu()

        if classes is None:
            classes = self.config['classes']

        if len(classes) != 2:
            raise ValueError('More than 2 classes found')

        if threshold is None:
            print(color('\nFinding optimal threshold based on: {}'.format(
                self.config['eval']['maximize_metric'])))
            maximize_fn = metric_factory.create(
                self.config['eval']['maximize_metric'],
                **{'recall': recall})
            _, _, threshold = maximize_fn(targets, predict_proba)

        predicted_labels = torch.ge(predict_proba, threshold)
        confusion_matrix = ConfusionMatrix(classes)
        confusion_matrix(targets, predicted_labels)

        # per class recall
        recall = Recall(reduction='none')(predicted_labels, targets)

        # standard metrics
        metrics = {
            'accuracy': Accuracy()(predicted_labels, targets),
            'confusion_matrix': confusion_matrix.cm,
            'threshold': float(threshold),

            # returns precision per class
            # take the values for class=1
            'precision': Precision(reduction='none')(
                predicted_labels, targets)[1],

            # recall = recall for class 1
            'recall': recall[1],

            # specificity = recall for class 0
            'specificity': recall[0],
        }

        # PR curve
        precisions, recalls, thresholds = PrecisionRecall()(
            predict_proba, targets)
        metrics['pr-curve'] = plot_classification_metric_curve(
            recalls, precisions, xlabel='Recall',
            ylabel='Precision')
        plt.close()

        # ROC curve
        fprs, tprs, _ = ROC()(predict_proba, targets)
        metrics['roc-curve'] = plot_classification_metric_curve(
            fprs, tprs, xlabel='False Positive Rate',
            ylabel='True Positive Rate')
        plt.close()

        # AUROC
        if len(torch.unique(targets)) == 1:
            # handle edge case
            metrics['auc-roc'] = 0
        else:
            metrics['auc-roc'] = AUROC()(predict_proba, targets)

        # Specificity-sensitivity curve
        specificities = np.array([1 - fpr for fpr in fprs])
        metrics['ss-curve'] = plot_classification_metric_curve(
            tprs, specificities, xlabel='Sensitivity',
            ylabel='Specificity')
        plt.close()

        return metrics

    def get_eval_params(
            self, predictions: torch.Tensor, targets: torch.Tensor,
            as_logits: bool = True) -> dict:
        """Get evaluation params by optimizing on the given data

        :param predictions: epoch predictions
        :type predictions: torch.Tensor
        :param targets: ground truths for the epoch
        :type targets: torch.Tensor
        :param as_logits: whether the predictions are logits; if
            as_logits=True, the values are converted into sigmoid scores
            before further processing.
        :type as_logits: bool, defaults to True

        :return: dict containing evaluation parameters
        """
        metrics = self.compute_epoch_metrics(predictions, targets)
        param_keys = ['recall', 'threshold']
        params = {key: metrics[key] for key in param_keys}
        return params

    def update_wandb(self, mode: str, epoch_outputs: dict, metrics: dict):
        """Logs values to wandb

        :param mode: train/val/test mode
        :type mode: str
        :param epoch_outputs: dictionary containing outputs for the epoch
        :type epoch_outputs: dict
        :param metrics: metrics for the epoch
        :type metrics: dict
        """
        super(BinaryClassificationModule, self).update_wandb(
                mode, epoch_outputs, metrics)

        if self.logger is not None:
            wandb_logs = {}

            wandb_logs[f'{mode}/confusion_matrix'] = wandb.Image(
                get_confusion_matrix(
                    metrics['confusion_matrix'], self.config['classes']))

            # log classification curves
            classification_curves = ['pr-curve', 'roc-curve', 'ss-curve']
            for curve in classification_curves:
                if curve in metrics:
                    wandb_logs[f'{mode}/{curve}'] = metrics[curve]

            # log to wandb
            self.logger.experiment.log(
                wandb_logs, step=self.logger.experiment.step)


class MultiClassClassificationModule(BinaryClassificationModule):
    """LightningModule for binary classification"""
    def compute_epoch_metrics(
            self, predictions: torch.Tensor, targets: torch.Tensor,
            threshold: float = None, recall: float = 0.9,
            as_logits: bool = True, classes: List[str] = None) -> dict:
        """Computes metrics for the epoch

        :param predictions: epoch predictions
        :type predictions: torch.Tensor
        :param targets: ground truths for the epoch
        :type targets: torch.Tensor
        :param threshold: confidence threshold to be used for binary
            classification; if None, the optimal threshold is found.
        :type threshold: float, defaults to None
        :param recall: minimum recall to choose the optimal threshold
        :type recall: float, defaults to 0.9
        :param as_logits: whether the predictions are logits; if
            as_logits=True, the values are converted into sigmoid scores
            before further processing.
        :type as_logits: bool, defaults to True
        :param classes: list of classes in the target
        :type classes: List[str], defaults to None

        :return: dictionary of metrics
        """
        targets = targets.cpu()
        predicted_labels = torch.argmax(predictions, dim=1).detach().cpu()

        if classes is None:
            classes = self.config['classes']

        confusion_matrix = ConfusionMatrix(classes)
        confusion_matrix(targets, predicted_labels)

        metrics = {
            'accuracy': Accuracy()(predicted_labels, targets),
            'confusion_matrix': confusion_matrix.cm,
        }

        return metrics

    def get_eval_params(
            self, predictions: torch.Tensor, targets: torch.Tensor,
            as_logits: bool = True) -> dict:
        """Get evaluation params by optimizing on the given data

        :param predictions: epoch predictions
        :type predictions: torch.Tensor
        :param targets: ground truths for the epoch
        :type targets: torch.Tensor
        :param as_logits: whether the predictions are logits; if
            as_logits=True, the values are converted into sigmoid scores
            before further processing.
        :type as_logits: bool, defaults to True

        :return: dict containing evaluation parameters
        """
        return {}
