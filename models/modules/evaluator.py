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
File: evaluator.py
"""
import os
import torch
import numpy as np
import cv2
import utils.tools as utils
from config import cf


class Evaluator:
    """
    Handles model test/evaluation functionality.
    """

    def __init__(self):

        # Report interval
        self.report_intv = 3
        self.results = []
        self.metadata = None

        # Make output and mask directories for results
        self.model_path = cf.load
        self.masks_dir = utils.mk_path(os.path.join(cf.output, 'masks'))
        self.output_dir = utils.mk_path(os.path.join(cf.output, 'outputs'))

    def load(self):

        """ load model file for evaluation"""

        if os.path.exists(self.model_path):
            return torch.load(self.model_path, map_location=cf.device)
        else:
            print('Model file: {} does not exist ... exiting.'.format(self.model_path))
            exit()

    def reset(self):
        self.results = []

    def save(self, fname):

        """Save full prediction test results with metadata for reconstruction"""
        # Build output file path
        output_file = os.path.join(self.output_dir, fname + '_output.pth')
        torch.save({"results": self.results, "metadata": self.metadata}, output_file)

    def save_image(self, fname):

        """Save prediction mask image"""

        # Build mask file path
        mask_file = os.path.join(self.masks_dir, fname + '.png')

        # Reconstruct seg-mask from predicted tiles
        tiles = np.concatenate(self.results, axis=0)
        mask_img = utils.reconstruct(tiles, self.metadata)

        # Save output mask image to file (RGB -> BGR conversion)
        # Note that the default color format in OpenCV is often
        # referred to as RGB but it is actually BGR (the bytes are reversed).
        cv2.imwrite(mask_file, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

        return mask_img
