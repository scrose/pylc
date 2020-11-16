"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Metrics Evaluator
File: metrics.py
"""
import numpy as np
from seaborn import heatmap, set
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


class Metrics:
    """
    Defines metrics to evaluate model outputs.
    Evaluate segmentation output accuracy
    """

    def __init__(self):

        # initialize matplotlib settings
        self.font = {'weight': 'bold', 'size': 18}
        self.plt = plt
        # plt.rc('font', **font)
        set(font_scale=0.9)

        # metrics evaluation results
        self.results = {}

        # single-image evaluation data buffers
        self.cmatrix = None
        self.cmap = None

    def report(self, y_true, y_pred, labels):
        """
        Generate Classification Report
        """
        self.results['report'] = classification_report(
            y_true,
            y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0
        )
        print('\nClassification Report')
        print(classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0
        ))

    def f1_score(self, y_true, y_pred):
        """ Compute Weighted F1 Score (DSC) """
        self.results['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print('{:30s}{}'.format('Weighted F1 Score', self.results['f1']))

    def jaccard(self, y_true, y_pred):
        """ Compute Weighted Jaccard (ioU) """
        self.results['iou'] = jaccard_score(y_true, y_pred, average='weighted')
        print('{:30s}{}'.format('Weighted IoU', self.results['iou']))

    def mcc(self, y_true, y_pred):
        """ Compute Matthews correlation coefficient """
        self.results['mcc'] = matthews_corrcoef(y_true, y_pred)
        print('{:30s}{}'.format('MCC', self.results['mcc']))

    def confusion_matrix(self, y_true, y_pred, labels):
        """
        Generate confusion matrix (Matplotlib).
        """
        self.cmatrix = confusion_matrix(y_true, y_pred, normalize='true')
        self.cmap = heatmap(
            self.cmatrix, vmin=0.01, vmax=1.0, fmt='.1g', xticklabels=labels, yticklabels=labels, annot=True)
        self.plt.ylabel('Ground-truth', fontsize=16, labelpad=6)
        self.plt.xlabel('Predicted', fontsize=16, labelpad=6)


def jsd(p, q):
    """
    Calculates Jensen–Shannon Divergence coefficient
    JSD measures the similarity between two probability
    distributions p and q.

      Parameters
      ------
      p: np.array array
         Probability distribution [n].
      q: np.array array
         Probability distribution [n].

      Returns
      ------
      float
         Computed JSD metric.
     """

    eps = 1e-8
    m = 0.5 * (p + q + eps)
    return 0.5 * np.sum(np.multiply(p, np.log(p / m + eps))) + 0.5 * np.sum(np.multiply(q, np.log(q / m + eps)))


def m2(p, n_classes):
    """
    Calculates the M2 Gibbs index that gives variance
    of a multinomial distribution

      Parameters
      ------
      p: np.array array
         Probability distribution [n].
      n_classes: int
         Number of classes.

      Returns
      ------
      float
         Computed M2 metric.
     """
    assert n_classes > 1, "M2 variance for multiple classes."
    return (n_classes / (n_classes - 1)) * (1 - np.sum(p ** 2))
