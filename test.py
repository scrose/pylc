"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
An evaluation of deep learning semantic segmentation for
land cover classification of oblique ground-based photography
MSc. Thesis 2020.
<http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

File: test.py
    Model testing.
"""

import os
import torch
from config import get_config
import utils.tools as utils
from utils.metrics import Metrics
from utils.extract import Extractor
from models.base import Model
from tqdm import trange
from params import params
import cv2
import numpy as np


def test(cf, model, bypass=False):
    """
    Apply model to input image(s) to generate segmentation maps.

    Parameters
    ------
    cf: dict
        User configuration settings.
    model: Model
        Network Mode (Pytorch).
    bypass: bool
        Argument parser.

     Returns
     ------
     bool
        Output boolean value.
    """

    # Initialize metrics evaluator
    metrics = Metrics(cf)

    # get test file(s)
    files = utils.collate(cf.img, cf.mask)

    # model test
    print('\nRunning model test ... ')

    # initialize extractor
    extractor = Extractor(cf)

    for f_idx, fpair in enumerate(files):

        # Check if future outputs exist already
        fid = os.path.basename(fpair.get('img')).replace('.', '_')
        output_file = os.path.join(cf.output, 'outputs', fid + '_output.pth')
        mask_file = os.path.join(cf.output, 'masks', fid + '.png')
        md_file = os.path.join(cf.output, 'outputs', fid + '_md.json')

        if os.path.exists(output_file) and input(
                "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(output_file)) != 'Y':
            print('Skipping')
            continue
        if os.path.exists(mask_file) and input(
                "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(mask_file)) != 'Y':
            print('Skipping')
            continue

        # extract tiles
        extractor.load(fpair.get('img'), fpair.get('mask')).extract(fit=True, stride=params.stride // 2, scale=cf.scale)

        # use default normalization if requested
        if cf.normalize_default:
            print(
                '\tInput normalized to default mean: {}, std: {}'.format(params.px_mean_default, params.px_std_default))

        # model.net.evaluate()
        n_samples = extractor.profiler.metadata['n_samples']

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


def main(cf):
    """
    Main model test handler

    Parameters
    ------
    cf: dict
        User configuration settings.
    """

    # Load pretrained model for testing or evaluation
    # Model file path is defined in user settings.
    model = Model(cf)
    # self.net = torch.nn.DataParallel(self.net)
    model.net.eval()

    test(cf, model)


if __name__ == "__main__":

    """ Parse model configuration settings. """
    config, unparsed, parser = get_config(params.TEST)

    # If we have unparsed arguments, or help request print usage and exit
    if len(unparsed) > 0 or config.h:
        parser.print_usage()
        exit()

    main(config)

# test.py ends here
