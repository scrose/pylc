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
from seaborn import heatmap, set
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import numpy as np
from params import params
import json
import utils.tex as tex


class Metrics:
    """
    Defines metrics to evaluate model outputs.
    Evaluate segmentation output accuracy

    Parameters:
    -----------
    y_true: numpy
        ground-truth data
    y_pred: numpy
        predicted data
    fid: str
        file ID

    Returns:
    --------
        Outputs accuracy metrics to file(s)
    """

    def __init__(self, cf, fid, y_true, y_pred):
        self.dpi = 400
        self.font = {'weight': 'bold', 'size': 18}
        plt.rc('font', **font)
        set(font_scale=0.9)
        # initialize metadata dict
        self.md = {
            'id': cf.id,
            'fid': fid
        }

        # Ensure true mask has all of the categories
        target_idx = np.unique(y_true)
        input_idx = np.unique(y_pred)
        label_idx = np.unique(np.concatenate((target_idx, input_idx)))

        self.labels = []
        # load category labels
        for idx in label_idx:
            self.labels += [params.schema(cf).labels[idx]]

        # Ensure true mask has all of the categories
        for idx in range(len(params.schema(cf).labels)):
            if idx not in target_idx:
                y_true[idx] = idx

        # store data
        self.y_true = y_true
        self.y_pred = y_pred

    def print_report(self):
        # Classification Report
        print(classification_report(self.y_true, self.y_pred, target_names=self.labels, zero_division=0))
        self.md['report'] = classification_report(
            self.y_true, self.y_pred, target_names=self.labels, output_dict=True, zero_division=0)

    def f1_score(self):
        # Weighted F1 Score (DSC)
        self.md['f1'] = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        print('Weighted F1 Score: {}'.format(self.md['f1']))

    def jaccard(self):
        # Weighted Jaccard (ioU)
        md['iou'] = jaccard_score(y_true, y_pred, average='weighted')
        print('Weighted IoU: {}'.format(md['iou']))

    def mcc(self):
        # Matthews correlation coefficient
        md['mcc'] = matthews_corrcoef(y_true, y_pred)
        print('MCC: {}'.format(md['mcc']))

    def confusion_matrix(self, y_true, y_pred, fid):
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
        conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
        cmap_path = os.path.join(cf.output_path, 'outputs', fid + '_cmap.pdf')
        # np.save(os.path.join(cf.output_path, 'outputs', fname + '_cmap.npy'), conf_matrix)
        cmap = heatmap(conf_matrix, vmin=0.01, vmax=1.0, fmt='.1g', xticklabels=labels, yticklabels=labels,
                       annot=True)
        plt.ylabel('Ground-truth', fontsize=16, labelpad=6)
        plt.xlabel('Predicted', fontsize=16, labelpad=6)
        cmap.get_figure().savefig(cmap_path, format='pdf', dpi=dpi)
        plt.clf()

    def save(self):
        # Save image metadata
        with open(os.path.join(cf.output_path, 'outputs', fid + '_md.json'), 'w') as fp:
            json.dump(md, fp, indent=4)

        # Save metadata as latex table
        # write back the new document
        with open(os.path.join(cf.output_path, 'outputs', fid + '_md.tex'), 'w') as fp:
            md_tex = tex.convert_md_to_tex(md)
            fp.write(md_tex)
