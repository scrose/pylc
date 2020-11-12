"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
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
from utils.tools import colourize


class Evaluator:
    """
    Handles model test/evaluation functionality.
    """

    def __init__(self, args):

        # initialize metrics
        self.params = Parameters(args)
        self.metrics = Metrics()

        # Model results
        self.fid = None
        self.logits = None
        self.mask_pred = None
        self.results = []
        self.md = {}

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
        self.output_dir = args.output_dir if hasattr(args, 'output_dir') else defaults.output_dir
        self.masks_dir = utils.mk_path(os.path.join(args.output_dir, 'masks'))
        self.logits_dir = utils.mk_path(os.path.join(args.output_dir, 'logits'))
        self.metrics_dir = utils.mk_path(os.path.join(args.output_dir, 'metrics'))

    def load(self, mask_pred_logits, meta, mask_true_path=None, scale=None):
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
        self.md = meta
        self.fid = self.md.extract['fid']

        # reconstruct unnormalized model outputs into mask data array
        self.logits = mask_pred_logits
        self.mask_pred = self.reconstruct()

        if mask_true_path:
            # load ground-truth data
            mask_true, w, h, w_scaled, h_scaled = utils.get_image(mask_true_path, 3, scale=scale, interpolate=cv2.INTER_NEAREST)

            # check dimensions of ground truth mask and predicted mask
            if not (w_scaled == self.md.extract['w_scaled'] and h_scaled == self.md.extract['h_scaled']):
                print("Ground truth mask dims ({}px X {}px) do not match predicted mask dims ({}px X {}px).".format(
                    w_scaled, h_scaled, self.md.extract['w_scaled'], self.md.extract['h_scaled']
                ))
                exit(1)

            self.y_true = torch.as_tensor(torch.tensor(mask_true), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
            self.y_pred = torch.as_tensor(self.mask_pred, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)

            # Class encode input predicted data
            self.y_pred = utils.class_encode(self.y_pred, self.md.palette_rgb)
            self.y_true = utils.class_encode(self.y_true, self.md.palette_rgb)

            # Verify same size of target == input
            assert self.y_pred.shape == self.y_true.shape, "Input dimensions {} not same as target {}.".format(
                self.y_pred.shape, self.y_true.shape)

            self.y_pred = self.y_pred.flatten()
            self.y_true = self.y_true.flatten()

            # load input data into metrics
            self.y_true_aggregate += [self.y_true]
            self.y_pred_aggregate += [self.y_pred]

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
        - Ensures all classes represented in ground truth mask
        """
        target_idx = np.unique(self.y_true)
        input_idx = np.unique(self.y_pred)
        label_idx = np.unique(np.concatenate((target_idx, input_idx)))

        # load category labels
        self.labels = []
        for idx in label_idx:
            self.labels += [self.params.class_codes[idx]]

        # Ensure true mask has all of the categories
        for idx in range(len(self.params.class_codes)):
            if idx not in target_idx:
                self.y_true[idx] = idx

        if self.aggregate:
            assert self.y_true_aggregate and self.y_pred_aggregate, \
                "Aggregate evaluation failed. Data buffer is empty."

            print("\nReporting aggregate metrics ... ")
            # Concatenate aggregated data
            self.y_true = np.concatenate((self.y_true_aggregate))
            self.y_true = np.concatenate((self.y_pred_aggregate))

        return self

    def reconstruct(self):
        """
        Reconstruct tiles into full-sized segmentation mask.
        Uses metadata generated from image tiling (adjust_to_tile)

          Returns
          ------
          mask_reconstructed: np.array
             Reconstructed image data.
         """

        # get tiles from tensor outputs
        tiles = np.concatenate((self.logits), axis=0)

        # load metadata
        w = self.md.extract['w_fitted']
        h = self.md.extract['h_fitted']
        w_full = self.md.extract['w_scaled']
        h_full = self.md.extract['h_scaled']
        offset = self.md.extract['offset']
        tile_size = self.md.tile_size
        stride = self.md.stride
        palette = self.md.palette_rgb
        n_classes = self.md.n_classes

        if stride < tile_size:
            n_strides_in_row = w // stride - 1
            n_strides_in_col = h // stride - 1
        else:
            n_strides_in_row = w // stride
            n_strides_in_col = h // stride

        # Calculate overlap
        olap_size = tile_size - stride

        # initialize full image numpy array
        mask_fullsized = np.empty((n_classes, h + offset, w), dtype=np.float32)

        # Create empty rows
        r_olap_prev = None
        r_olap_merged = None

        # row index (set to offset height)
        row_idx = offset

        for i in range(n_strides_in_col):
            # Get initial tile in row
            t_current = tiles[i * n_strides_in_row]
            r_current = np.empty((n_classes, tile_size, w), dtype=np.float32)
            col_idx = 0
            # Step 1: Collate column tiles in row
            for j in range(n_strides_in_row):
                t_current_width = t_current.shape[2]
                if j < n_strides_in_row - 1:
                    # Get adjacent tile
                    t_next = tiles[i * n_strides_in_row + j + 1]
                    # Extract right overlap of current tile
                    olap_current = t_current[:, :, t_current_width - olap_size:t_current_width]
                    # Extract left overlap of next (adjacent) tile
                    olap_next = t_next[:, :, 0:olap_size]
                    # Average the overlapping segment logits
                    olap_current = torch.nn.functional.softmax(torch.tensor(olap_current), dim=0)
                    olap_next = torch.nn.functional.softmax(torch.tensor(olap_next), dim=0)
                    olap_merged = (olap_current + olap_next) / 2
                    # Insert averaged overlap into current tile
                    np.copyto(t_current[:, :, t_current_width - olap_size:t_current_width], olap_merged)
                    # Insert updated current tile into row
                    np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)
                    col_idx += t_current_width
                    # Crop next tile and copy to current tile
                    t_current = t_next[:, :, olap_size:t_next.shape[2]]

                else:
                    np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)

            # Step 2: Collate row slices into full mask
            r_current_height = r_current.shape[1]
            # Extract overlaps at top and bottom of current row
            r_olap_top = r_current[:, 0:olap_size, :]
            r_olap_bottom = r_current[:, r_current_height - olap_size:r_current_height, :]

            # Average the overlapping segment logits
            if i > 0:
                # Average the overlapping segment logits
                r_olap_top = torch.nn.functional.softmax(torch.tensor(r_olap_top), dim=0)
                r_olap_prev = torch.nn.functional.softmax(torch.tensor(r_olap_prev), dim=0)
                r_olap_merged = (r_olap_top + r_olap_prev) / 2

            # Top row: crop by bottom overlap (to be averaged)
            if i == 0:
                # Crop current row by bottom overlap size
                r_current = r_current[:, 0:r_current_height - olap_size, :]
            # Otherwise: Merge top overlap with previous
            else:
                # Replace top overlap with averaged overlap in current row
                np.copyto(r_current[:, 0:olap_size, :], r_olap_merged)

            # Crop middle rows by bottom overlap
            if 0 < i < n_strides_in_col - 1:
                r_current = r_current[:, 0:r_current_height - olap_size, :]

            # Copy current row to full mask
            np.copyto(mask_fullsized[:, row_idx:row_idx + r_current.shape[1], :], r_current)
            row_idx += r_current.shape[1]
            r_olap_prev = r_olap_bottom

        # colourize to palette
        mask_fullsized = np.expand_dims(mask_fullsized, axis=0)
        _mask_pred = colourize(np.argmax(mask_fullsized, axis=1), n_classes, palette=palette)

        # resize mask to full size
        mask_reconstructed = cv2.resize(
            _mask_pred[0].astype('float32'), (w_full, h_full), interpolation=cv2.INTER_NEAREST)

        return mask_reconstructed

    def reset(self):
        """
        Resets evaluator buffers.
        """
        self.logits = None
        self.mask_pred = None
        self.results = []
        self.md = {}
        self.y_true = None
        self.y_pred = None
        self.y_true_aggregate = []
        self.y_pred_aggregate = []

    def save_logits(self):
        """
        Save unnormalized model outputs (logits) to file.

        Returns
        -------
        logits_file: str
            Output path to model outputs file.
        """
        # save unnormalized model outputs
        logits_file = os.path.join(self.logits_dir, self.fid + '_output.pth')
        if utils.confirm_write_file(logits_file):
            torch.save({"results": self.results, "meta": self.md}, logits_file)
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
                fp.write(tex.convert_md_to_tex(self.md))
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
