"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
An evaluation of deep learning semantic segmentation for
land cover classification of oblique ground-based photography
MSc. Thesis 2020.
<http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Model Test
File: test.py
"""

import os
import torch
import utils.tools as utils
from utils.metrics import Metrics
from utils.extract import extractor
from models.model import Model
from tqdm import trange
import cv2
import numpy as np
from config import cf


def test():
    """
    Apply model to input image(s) to generate segmentation maps.
    """

    # Load pretrained model for testing or evaluation
    # Model file path is defined in user settings.
    model = Model()
    # self.net = torch.nn.DataParallel(self.net)
    model.net.eval()

    # Initialize metrics evaluator
    metrics = Metrics()

    # get test file(s) - returns list of filenames
    files = utils.collate(cf.img, cf.mask)

    # model test
    print('\nRunning model test ... ')

    for f_idx, fpair in enumerate(files):

        # get image and associated mask data
        if type(fpair) == dict and fpair.has_key('img') and fpair.has_key('mask'):
            img_file = fpair.get('img')
            mask_file = fpair.get('mask')
        else:
            img_file = fpair
            mask_file = None

        # initialize output file paths (use cf.output directory)
        fid = os.path.basename(fpair.get('img')).replace('.', '_')
        output_file = os.path.join(cf.output, 'outputs', fid + '_output.pth')
        if utils.confirm_write_file(output_file):
            continue
        metrics_file = os.path.join(cf.output, 'outputs', fid + '_evaluation.json')
        if utils.confirm_write_file(metrics_file):
            continue
        output_mask_file = os.path.join(cf.output, 'masks', fid + '.png')
        if utils.confirm_write_file(output_mask_file):
            continue

        # extract image tiles
        img_tiles = extractor.load(img_file).extract(fit=True, stride=cf.tile_size//2)

        # use default normalization if requested
        if cf.normalize_default:
            print(
                '\tInput normalized to default mean: {}, std: {}'.format(cf.px_mean_default, cf.px_std_default))

        # apply model to input
        with torch.no_grad():
            for i in trange(n_samples):
                x = extractor.imgs[i].unsqueeze(0).float()
                model.test(x)
                model.iter += 1

                # Save prediction test segmentation output (logits) to file
                if cf.save_output:
                    # copy extraction details to evaluator
                    model.evaluator.metadata = extractor.profiler.metadata['extraction']
                    model.evaluator.save(fid)
                    print("Output data saved to {}.".format(model.evaluator.output_path))

                # Save full mask image to file
                y_pred = model.evaluator.save_image(fid)
                print("Output mask saved to {}.".format(model.evaluator.masks_path))

        # If ground truth masks provided, evaluate segmentation accuracy
        if cf.mask:

            if not cf.global_metrics and os.path.exists(md_file) and input(
                    "\tEvaluation metrics file {} exists. Re-do evaluation? (Type \'Y\' for yes): ".format(
                        md_file)) != 'Y':
                continue

            # Evaluate prediction against ground-truth
            if not cf.global_metrics:
                print("\nStarting evaluation of outputs ... ")
                metrics.load(fid, cf.mask, mask_file).evaluate()

        # Reset evaluator
        model.evaluator.reset()

    # Compute global metrics
    if cf.global_metrics:
        metrics.evaluate(aggregate=True)


def reconstruct(config):
    """
    -------------------------------------
     Reconstruct mask from output segmentation tiles
    -------------------------------------
     Inputs:     configuration settings (dict)
     Outputs:    Reconstructed segmentation mask
    -------------------------------------
    """

    # load output data
    output = torch.load(config.output_path, map_location=lambda storage, loc: storage)
    print('Loaded results for {}'.format(config.output_path))

    # get unary output data / metadata
    if type(output['results']) == list:
        unary_data = np.concatenate(output['results'])
    else:
        unary_data = output['results'].numpy()

    md = output['metadata']

    # Reconstruct seg-mask from predicted tiles
    mask_img = utils.reconstruct(unary_data, md)

    # Extract image path
    fid = os.path.basename(config.output_path).replace('.', '_')

    mask_file = os.path.join(config.mask_path, fid + '.png')

    # Save output mask image to file (RGB -> BGR conversion)
    # Note that the default color format in OpenCV is often
    # referred to as RGB but it is actually BGR (the bytes are reversed).
    cv2.imwrite(mask_file, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

    print('Reconstructed mask saved to {}'.format(mask_file))


# test.py ends here
