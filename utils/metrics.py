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

import os
import json
import numpy as np
import torch
from seaborn import heatmap, set
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import utils.tex as tex
import utils.tools as utils
from params import params


class Metrics:
    """
    Defines metrics to evaluate model outputs.
    Evaluate segmentation output accuracy

    Returns:
    --------
        Outputs accuracy metrics to file(s)
    """

    def __init__(self, cf):
        self.dpi = 400
        self.font = {'weight': 'bold', 'size': 18}
        # plt.rc('font', **font)
        set(font_scale=0.9)

        # metrics metadata
        self.output_path = cf.output
        self.md = {
            'id': cf.id
        }
        self.schema = params.settings.schemas[cf.schema]
        self.labels = []
        self.fid = None
        self.y_true = None
        self.y_pred = None

        # global metrics available for multiple computations
        self.y_true_global = None
        self.y_pred_global = None

    def load(self, fid, mask_true, mask_pred):
        """
        Initialize predicted/ground truth image masks for
        evaluation metrics.

        Parameters:
        -----------
        mask_true: numpy
            ground-truth mask image [CHW]
        mask_pred: numpy
            predicted mask image [CHW]
        fid: str
            file ID
        """
        self.fid = fid
        self.md.update({'fid': fid, 'mask_true': mask_true, 'mask_pred': mask_pred})

        # load ground-truth data
        y_true = torch.as_tensor(utils.get_image(mask_true, 3), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
        y_pred = torch.as_tensor(utils.get_image(mask_pred, 3), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)

        # Class encode input predicted data
        y_pred = utils.class_encode(y_pred, self.schema.palette)
        y_true = utils.class_encode(y_true, self.schema.palette)

        # Verify same size of target == input
        assert y_pred.shape == y_true.shape, "Input dimensions {} not same as target {}.".format(
            y_pred.shape, y_true.shape)

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        self.y_true_global += [y_true]
        self.y_pred_global += [y_pred]

        self.y_true = y_true
        self.y_pred = y_pred

        return self

    def validate(self):
        """ Ensure true mask has all of the categories """

        target_idx = np.unique(self.y_true)
        input_idx = np.unique(self.y_pred)
        label_idx = np.unique(np.concatenate((target_idx, input_idx)))

        self.labels = []
        # load category labels
        for idx in label_idx:
            self.labels += [self.schema.labels[idx]]

        # Ensure true mask has all of the categories
        for idx in range(len(self.schema.labels)):
            if idx not in target_idx:
                self.y_true[idx] = idx

        return self

    def evaluate(self, aggregate=False):
        """
        Compute evaluation metrics and save to file

        Parameters
        ----------
        aggregate: bool
            Compute aggregate metrics for multiple data loads.
        """
        if aggregate:
            print("\nReporting global metrics ... ")
            # Concatenate aggregated data
            self.y_true = np.concatenate((self.y_true_global))
            self.y_true = np.concatenate((self.y_pred_global))

        self.f1_score()
        self.jaccard()
        self.mcc()
        self.confusion_matrix()
        self.print_report()
        self.save()

        return self

    def print_report(self):
        """ Generate Classification Report """
        print(classification_report(self.y_true, self.y_pred, target_names=self.labels, zero_division=0))
        self.md['report'] = classification_report(
            self.y_true, self.y_pred, target_names=self.labels, output_dict=True, zero_division=0)

    def f1_score(self):
        """ Compute Weighted F1 Score (DSC) """
        self.md['f1'] = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        print('Weighted F1 Score: {}'.format(self.md['f1']))

    def jaccard(self):
        """ Compute Weighted Jaccard (ioU) """
        self.md['iou'] = jaccard_score(self.y_true, self.y_pred, average='weighted')
        print('Weighted IoU: {}'.format(self.md['iou']))

    def mcc(self):
        """ Compute Matthews correlation coefficient """
        self.md['mcc'] = matthews_corrcoef(self.y_true, self.y_pred)
        print('MCC: {}'.format(self.md['mcc']))

    def reset(self):
        """ Reset metric properties """
        self.y_true = None
        self.y_pred = None
        self.y_true_global = None
        self.y_pred_global = None

    def confusion_matrix(self):
        """
        Generate confusion matrix.

        Parameters:
        -----------
        cf: dict
            configuration settings
        y_true: numpy
            ground-truth data
        y_pred: numpy
            predicted data
        fid: str
            file ID
         Returns:
         --------
         Saves accuracy metrics to file(s)
        """
        conf_matrix = confusion_matrix(self.y_true, self.y_pred, normalize='true')
        cmap_path = os.path.join(self.output_path, 'outputs', self.fid + '_cmap.pdf')
        # np.save(os.path.join(cf.output_path, 'outputs', fname + '_cmap.npy'), conf_matrix)
        cmap = heatmap(conf_matrix, vmin=0.01, vmax=1.0, fmt='.1g', xticklabels=self.labels, yticklabels=self.labels,
                       annot=True)
        plt.ylabel('Ground-truth', fontsize=16, labelpad=6)
        plt.xlabel('Predicted', fontsize=16, labelpad=6)
        cmap.get_figure().savefig(cmap_path, format='pdf', dpi=self.dpi)
        plt.clf()

    def save(self):
        # Save image metadata
        with open(os.path.join(self.output_path, 'outputs', self.fid + '_md.json'), 'w') as fp:
            json.dump(self.md, fp, indent=4)

        # Save metadata as latex table
        # write back the new document
        with open(os.path.join(self.output_path, 'outputs', self.fid + '_md.tex'), 'w') as fp:
            md_tex = tex.convert_md_to_tex(self.md)
            fp.write(md_tex)
