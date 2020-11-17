"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Evaluator Class
File: evaluate.py
"""
import json
import os
import torch
import utils.tex as tex
import numpy as np
import cv2
import utils.tools as utils
from config import defaults, Parameters
from utils.metrics import Metrics


class Evaluator:
    """
    Handles model test/evaluation functionality.

    Parameters
    ------
    params: Parameters
        Updated parameters.
    """

    def __init__(self, params=None):

        # initialize parameters, metrics
        self.meta = Parameters(params) if params is not None else defaults
        self.metrics = Metrics()

        # Model results
        self.fid = None
        self.logits = None
        self.mask_pred = None
        self.results = []

        # data buffers
        self.y_true = None
        self.y_pred = None
        self.labels = []

        # multi-image data buffers for aggregate evaluation
        self.aggregate = False
        self.y_true_aggregate = []
        self.y_pred_aggregate = []

        # Make output and mask directories for results
        self.model_path = None
        self.output_dir = os.path.join(defaults.output_dir, self.meta.id)
        self.masks_dir = utils.mk_path(os.path.join(self.output_dir, 'masks'))
        self.logits_dir = utils.mk_path(os.path.join(self.output_dir, 'logits'))
        self.metrics_dir = utils.mk_path(os.path.join(self.output_dir, 'metrics'))

    def load(self, mask_pred, meta, mask_true_path=None, scale=None):
        """
        Initialize predicted/ground truth image masks for
        evaluation metrics.

        Parameters:
        -----------
        mask_pred_logits: torch.tensor
            Unnormalized model logits for predicted segmentation [NCHW]
        meta: dict
            Reconstruction metadata.
        mask_true_path: str
            File path to ground-truth mask [CHW]
        """

        # store metadata
        self.meta = meta
        # file identifier (include current scale)
        self.fid = self.meta.extract['fid']

        # reconstruct unnormalized model outputs into mask data array
        self.mask_pred = mask_pred

        if mask_true_path:
            # load ground-truth data
            mask_true, w, h, w_scaled, h_scaled = utils.get_image(
                mask_true_path,
                ch=3,
                scale=scale,
                interpolate=cv2.INTER_NEAREST
            )

            # check dimensions of ground truth mask and predicted mask
            if not (w_scaled == self.meta.extract['w_scaled'] and h_scaled == self.meta.extract['h_scaled']):
                print("Ground truth mask dims ({}px X {}px) do not match predicted mask dims ({}px X {}px).".format(
                    w_scaled, h_scaled, self.meta.extract['w_scaled'], self.meta.extract['h_scaled']
                ))
                exit(1)

            self.y_true = torch.as_tensor(torch.tensor(mask_true), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
            self.y_pred = torch.as_tensor(self.mask_pred, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)

            # Class encode input predicted data
            self.y_pred = utils.class_encode(self.y_pred, self.meta.palette_rgb)
            self.y_true = utils.class_encode(self.y_true, self.meta.palette_rgb)

            # Verify same size of target == input
            assert self.y_pred.shape == self.y_true.shape, "Input dimensions {} not same as target {}.".format(
                self.y_pred.shape, self.y_true.shape)

            self.y_pred = self.y_pred.flatten()
            self.y_true = self.y_true.flatten()

            # load input data into metrics
            self.y_true_aggregate += [self.y_true]
            self.y_pred_aggregate += [self.y_pred]

        return self

    def update(self, meta):
        """
        Update local metadata

        """
        self.meta = meta
        return self

    def evaluate(self, aggregate=False):
        """
        Compute evaluation metrics

        Parameters
        ----------
        aggregate: bool
            Compute aggregate metrics for multiple data loads.
        """
        self.aggregate = aggregate
        self.validate()
        self.metrics.f1_score(self.y_true, self.y_pred)
        self.metrics.jaccard(self.y_true, self.y_pred)
        self.metrics.mcc(self.y_true, self.y_pred)
        self.metrics.confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        self.metrics.report(self.y_true, self.y_pred, labels=self.labels)

        return self

    def validate(self):
        """
        Validates mask data for computations.
        - Ensures all classes are represented in ground truth mask.
        """

        self.labels = defaults.class_codes

        # aggregated metrics
        if self.aggregate:
            self.fid = 'aggregate_metrics'
            assert self.y_true_aggregate and self.y_pred_aggregate, \
                "Aggregate evaluation failed. Data buffer is empty."

            print("\nReporting aggregate metrics ... ")
            print("\t - Total generated masks: {}".format(len(self.y_pred_aggregate)))
            print()
            # Concatenate aggregated data
            self.y_true = np.concatenate((self.y_true_aggregate))
            self.y_pred = np.concatenate((self.y_pred_aggregate))

        # ensure class coverage
        for idx in range(len(self.labels)):
            self.y_true[idx] = idx
            self.y_pred[idx] = idx

        return self

    def reset(self):
        """
        Resets evaluator buffers.
        """
        self.logits = None
        self.mask_pred = None
        self.results = []
        self.meta = {}
        self.y_true = None
        self.y_pred = None

    def save_logits(self, logits):
        """
        Save unnormalized model outputs (logits) to file.

        Parameters
        ----------
        logits: list
            Unnormalized model outputs.
        Returns
        -------
        logits_file: str
            Output path to model outputs file.
        """
        # save unnormalized model outputs
        logits_file = os.path.join(self.logits_dir, self.fid + '_output.pth')
        if utils.confirm_write_file(logits_file):
            torch.save({"results": logits, "meta": self.meta}, logits_file)
            print("Model output data saved to \n\t{}.".format(logits_file))
            return logits_file
        return

    def save_metrics(self):
        """
        Save prediction evaluation results to files.

        Returns
        -------
        metrics_file: str
            Output path to metrics data file.
        metrics_file: str
            Output path to confusion matrix PDF file.
        metrics_file: str
            Output path to confusion matrix data file.
        """
        # Build output file paths
        metrics_file = os.path.join(self.metrics_dir, self.fid + '_eval.json')
        cmap_img_file = os.path.join(self.metrics_dir, self.fid + '_cmap.pdf')
        cmap_data_file = os.path.join(self.metrics_dir, self.fid + '_cmap.npy')

        # save evaluation metrics results as JSON file
        if utils.confirm_write_file(metrics_file):
            with open(metrics_file, 'w') as fp:
                json.dump(self.metrics.results, fp, indent=4)
        # save confusion matrix as PDF and data file
        if utils.confirm_write_file(cmap_img_file):
            self.metrics.cmap.get_figure().savefig(cmap_img_file, format='pdf', dpi=400)
            np.save(cmap_data_file, self.metrics.cmatrix)

        # clear metrics plot
        self.metrics.plt.clf()
        return metrics_file, cmap_img_file, cmap_data_file

    def save_tex(self):
        """
        Save prediction evaluation results as LaTeX table to file.

        Returns
        -------
        tex_file: str
            Output path to TeX data file.
        """
        tex_file = os.path.join(self.metrics_dir, self.fid + '_metrics.tex')
        if utils.confirm_write_file(tex_file):
            with open(tex_file, 'w') as fp:
                fp.write(tex.convert_md_to_tex(self.meta))
                return tex_file
        return

    def save_image(self):
        """
        Reconstructs segmentation prediction as mask image.
        Output mask image saved to file (RGB -> BGR conversion)
        Note that the default color format in OpenCV is often
        referred to as RGB but it is actually BGR (the bytes are
        reversed).

        Returns
        -------
        mask_data: np.array
            Output mask data.
        """

        # Build mask file path
        mask_file = os.path.join(self.masks_dir, self.fid + '.png')

        if self.mask_pred is None:
            print("Mask has not been reconstructed. Image save cancelled.")

        if utils.confirm_write_file(mask_file):
            # Reconstruct seg-mask from predicted tiles and write to file
            cv2.imwrite(mask_file, cv2.cvtColor(self.mask_pred, cv2.COLOR_RGB2BGR))

            print("Output mask saved to: \n\t{}.".format(mask_file))

            return mask_file
        return
