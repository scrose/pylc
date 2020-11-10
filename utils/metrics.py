"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
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
from utils.tools import get_schema


class Metrics:
    """
    Defines metrics to evaluate model outputs.
    Evaluate segmentation output accuracy

    Parameters
    ------
    args: dict
        User-defined configuration settings.
    """

    def __init__(self, args):
        # initialize matplotlib settings
        self.font = {'weight': 'bold', 'size': 18}
        self.plt = plt
        # plt.rc('font', **font)
        set(font_scale=0.9)

        # metrics metadata
        self.meta = None
        self.labels = []
        self.fid = None

        # get schema palettes, labels, categories
        schema = get_schema(args.schema)
        self.class_labels = schema.class_labels
        self.class_codes = schema.class_codes
        self.palette_hex = schema.palette_hex
        self.palette_rgb = schema.palette_rgb
        self.n_classes = schema.n_classes

        # single-image evaluation data buffers
        self.y_true = None
        self.y_pred = None
        self.cmatrix = None
        self.cmap = None

        # multi-image data buffers for aggregate evaluation
        self.y_true_aggregate = None
        self.y_pred_aggregate = None

    def validate(self):
        """
        Validates mask data for computations.
        - Ensures all classes represented in ground truth mask
        """
        target_idx = np.unique(self.y_true)
        input_idx = np.unique(self.y_pred)
        label_idx = np.unique(np.concatenate((target_idx, input_idx)))

        # load category labels
        self.labels = []
        for idx in label_idx:
            self.labels += [self.class_labels[idx]]

        # Ensure true mask has all of the categories
        for idx in range(len(self.class_labels)):
            if idx not in target_idx:
                self.y_true[idx] = idx

        return self

    def evaluate(self, aggregate=False):
        """
        Compute evaluation metrics

        Parameters
        ----------
        aggregate: bool
            Compute aggregate metrics for multiple data loads.
        """

        assert self.y_true_aggregate and self.y_true_aggregate, "Global evaluation failed. Data buffer is empty."

        if aggregate:
            print("\nReporting global metrics ... ")
            # Concatenate aggregated data
            self.y_true = np.concatenate((self.y_true_aggregate))
            self.y_true = np.concatenate((self.y_pred_aggregate))

        self.f1_score()
        self.jaccard()
        self.mcc()
        self.confusion_matrix()
        self.report()

        return self

    def report(self):
        """ Generate Classification Report """
        print(classification_report(self.y_true, self.y_pred, target_names=self.labels, zero_division=0))
        self.meta['report'] = classification_report(
            self.y_true, self.y_pred, target_names=self.labels, output_dict=True, zero_division=0)

    def f1_score(self):
        """ Compute Weighted F1 Score (DSC) """
        self.meta['f1'] = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        print('Weighted F1 Score: {}'.format(self.meta['f1']))

    def jaccard(self):
        """ Compute Weighted Jaccard (ioU) """
        self.meta['iou'] = jaccard_score(self.y_true, self.y_pred, average='weighted')
        print('Weighted IoU: {}'.format(self.meta['iou']))

    def mcc(self):
        """ Compute Matthews correlation coefficient """
        self.meta['mcc'] = matthews_corrcoef(self.y_true, self.y_pred)
        print('MCC: {}'.format(self.meta['mcc']))

    def reset(self):
        """ Reset metric properties """
        self.y_true = None
        self.y_pred = None
        self.y_true_aggregate = None
        self.y_pred_aggregate = None

    def confusion_matrix(self):
        """
        Generate confusion matrix (Matplotlib).
        """
        self.cmatrix = confusion_matrix(self.y_true, self.y_pred, normalize='true')
        self.cmap = heatmap(
            self.cmatrix, vmin=0.01, vmax=1.0, fmt='.1g', xticklabels=self.labels, yticklabels=self.labels, annot=True)
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