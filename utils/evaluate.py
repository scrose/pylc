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
from utils.metrics import Metrics
from utils.tools import colourize


class Evaluator:
    """
    Handles model test/evaluation functionality.
    """

    def __init__(self, args):

        # initialize metrics
        self.metrics = Metrics(args)

        # Model results
        self.fid = None
        self.logits = None
        self.mask_pred = None
        self.results = []
        self.meta = {}

        # Make output and mask directories for results
        self.output_dir = args.output
        self.model_path = None
        self.masks_dir = utils.mk_path(os.path.join(self.output_dir, 'masks'))
        self.logits_dir = utils.mk_path(os.path.join(self.output_dir, 'logits'))
        self.metrics_dir = utils.mk_path(os.path.join(self.output_dir, 'metrics'))

    def load(self, model_outputs, meta, target=None):
        """
        Initialize predicted/ground truth image masks for
        evaluation metrics.

        Parameters:
        -----------
        mask_true: np.array
            ground-truth mask image [CHW]
        model_outputs: np.array
            Unnormalized model logits for predicted segmentation [NCHW]
        fid: str
            file ID
        """
        # store metadata
        self.meta = meta
        self.fid = meta.extract['fid']

        # reconstruct unnormalized model outputs into mask data array
        self.logits = model_outputs
        self.mask_pred = self.reconstruct()

        if target:

            # load ground-truth data
            y_true = torch.as_tensor(utils.get_image(target, 3),
                                     dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
            y_pred = torch.as_tensor(utils.get_image(self.mask_pred, 3),
                                     dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)

            # Class encode input predicted data
            y_pred = utils.class_encode(y_pred, self.metrics.palette_rgb)
            y_true = utils.class_encode(y_true, self.metrics.palette_rgb)

            # Verify same size of target == input
            assert y_pred.shape == y_true.shape, "Input dimensions {} not same as target {}.".format(
                y_pred.shape, y_true.shape)

            y_pred = y_pred.flatten()
            y_true = y_true.flatten()

            # load input data into metrics
            self.metrics.y_true_aggregate += [y_true]
            self.metrics.y_pred_aggregate += [y_pred]
            self.metrics.y_true = y_true
            self.metrics.y_pred = y_pred

        return self

    def reset(self):
        """
        Resets evaluator buffers.
        """
        self.logits = None
        self.mask_pred = None
        self.results = []
        self.meta = {}

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
            torch.save({"results": self.results, "meta": self.meta}, logits_file)
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

        # save evaluation metrics as JSON file
        if utils.confirm_write_file(metrics_file):
            with open(metrics_file, 'w') as fp:
                json.dump(self.meta, fp, indent=4)
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

        if utils.confirm_write_file(mask_file):
            # Reconstruct seg-mask from predicted tiles and write to file
            mask_img = self.reconstruct()
            cv2.imwrite(mask_file, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

            print("Output mask saved to: \n\t{}.".format(mask_file))

            return mask_file
        return

    def reconstruct(self):
        """
        Reconstruct tiles into full-sized segmentation mask.
        Uses metadata generated from image tiling (adjust_to_tile)

          Returns
          ------
          mask_reconstructed: np.array
             Reconstructed image data.
         """

        # load metadata
        # self.results = np.concatenate(self.results, axis=0)
        w = self.meta['w']
        h = self.meta['h']
        w_full = self.meta['w_full']
        h_full = self.meta['h_full']
        offset = self.meta['offset']
        stride = self.meta['stride']
        palette = self.meta['palette']
        n_classes = self.meta['n_classes']

        # Calculate reconstruction dimensions
        tile_size = self.logits.shape[2]

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
            t_current = self.logits[i * n_strides_in_row]
            r_current = np.empty((n_classes, tile_size, w), dtype=np.float32)
            col_idx = 0
            # Step 1: Collate column tiles in row
            for j in range(n_strides_in_row):
                t_current_width = t_current.shape[2]
                if j < n_strides_in_row - 1:
                    # Get adjacent tile
                    t_next = self.logits[i * n_strides_in_row + j + 1]
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

        # Colourize and resize mask to full size
        mask_fullsized = np.expand_dims(mask_fullsized, axis=0)
        _mask_pred = colourize(np.argmax(mask_fullsized, axis=1), n_classes, palette=palette)
        mask_reconstructed = cv2.resize(
            _mask_pred[0].astype('float32'), (w_full, h_full), interpolation=cv2.INTER_NEAREST)

        return mask_reconstructed

