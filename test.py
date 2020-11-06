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
from utils.extractor import Extractor
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

    # Initialize files list
    y_true_overall = []
    y_pred_overall = []

    # get test file(s)
    files = utils.collate(cf.img, cf.mask)

    # model test
    print('\nRunning model test ... ')

    # create extractor to extract tiles
    extractor = Extractor(cf).load(cf.img, cf.mask).extract()


    for f_idx, fpair in enumerate(files):
        # Get image and associated target data
        img_path = fpair.get('img')
        mask_path = fpair.get('mask')
        y_pred = None
        get_output = True




        # Check if output exists already
        fname = os.path.basename(img_path).replace('.', '_')
        output_file = os.path.join(cf.output, 'outputs', fname + '_output.pth')
        md_file = os.path.join(cf.output_path, 'outputs', fname + '_md.json')
        mask_file = os.path.join(cf.output_path, 'masks', fname + '.png')


        if os.path.exists(output_file) and input(
                "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(output_file)) != 'Y':
            get_output = False
        if os.path.exists(mask_file) and input(
                "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(mask_file)) != 'Y':
            get_output = False

            if get_output:
                # Extract image subimages [NCWH format]
                img = utils.get_image(img_path, cf.in_channels)
                w_full = img.shape[1]
                h_full = img.shape[0]

                print('\n---\nTest Image [{}]: {}'.format(f_idx, img_path))
                print('\tWidth: {}px'.format(w_full))
                print('\tHeight: {}px'.format(h_full))
                print('\tChannels: {}'.format(cf.in_channels))

                if cf.normalize_default:
                    print('\tInput normalized to default mean: {}, std: {}'.format(params.px_mean_default, params.px_std_default))

                if cf.resample:
                    img = utils.get_image(cf.img_path, cf.in_channels, scale=cf.resample)
                    w_full = img.shape[1]
                    h_full = img.shape[0]

                    print('\n---\nResampled:')
                    print('\tScaling: {}'.format(cf.resample))
                    print('\tWidth: {}px'.format(w_full))
                    print('\tHeight: {}px'.format(h_full))

                # Set stride to half tile size
                stride = params.patch_size // 2

                # Adjust image size to fit N tiles
                img, w, h, offset = utils.adjust_to_tile(img, params.patch_size, stride, cf.in_channels)
                print('\nImage Resized to: ')
                print('\tWidth: {}px'.format(w))
                print('\tHeight: {}px'.format(h))
                print('\tTop offset: {}'.format(offset))

                # Convert image to tensor
                img_data = torch.as_tensor(img, dtype=torch.float32)

                # Create image tiles
                print("\nCreating test image tiles ... ")
                img_data = img_data.unfold(0, params.patch_size, stride).unfold(1, params.patch_size, stride)
                img_data = torch.reshape(img_data,
                                         (img_data.shape[0] * img_data.shape[1], cf.in_channels, params.patch_size,
                                          params.patch_size))
                n_samples = int(img_data.shape[0] * cf.clip)

                print('\nImage Tiles: ')
                print('\tN: {}'.format(img_data.shape[0]))
                print('\tSize: {}px'.format(img_data.shape[2]))
                print('\tChannels: {}'.format(img_data.shape[1]))
                print('\tStride: {}'.format(stride))

                model.evaluator.metadata = {
                    "w": w,
                    "h": h,
                    "w_full": w_full,
                    "h_full": h_full,
                    "offset": offset,
                    "stride": stride,
                    "n_samples": n_samples
                }

                # model.net.evaluate()

                print('\nProcessing image tiles ... ')
                with torch.no_grad():
                    for i in trange(n_samples):
                        x = img_data[i].unsqueeze(0).float()
                        model.test(x)
                        model.iter += 1

                # Save prediction test output to file
                if cf.save_output:
                    model.evaluator.save(fname)
                    print("Output data saved to {}.".format(model.evaluator.output_path))

                # Save full mask image to file
                y_pred = model.evaluator.save_image(fname)
                print("Output mask saved to {}.".format(model.evaluator.masks_path))

        # ==============================
        # Evaluation segmentation accuracy (requires ground-truth dataset)
        # ==============================
        if cf.validate:
            if not cf.global_metrics and os.path.exists(md_file) and input(
                    "\tMetadata file {} exists. Re-do validation? (Type \'Y\' for yes): ".format(md_file)) != 'Y':
                continue
            # load ground-truth data
            print("\nStarting evaluation of outputs ... ")
            y_true = torch.as_tensor(utils.get_image(mask_path, 3), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
            print("\tLoading mask file {}".format(mask_file))
            y_pred = torch.as_tensor(utils.get_image(mask_file, 3), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)

            # Class encode input predicted data
            y_pred = utils.class_encode(y_pred, params.palette_lcc_a)
            y_true = utils.class_encode(y_true, params.palette_lcc_a)

            # Verify same size of target == input
            assert y_pred.shape == y_true.shape, "Input dimensions {} not same as target {}.".format(
                y_pred.shape, y_true.shape)

            # Flatten data for analysis
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()

            y_true_overall += [y_true]
            y_pred_overall += [y_pred]

            # Evaluate prediction against ground-truth
            if not cf.global_metrics:
                evaluate(conf, y_true, y_pred, fname)

        # Reset evaluator
        model.evaluator.reset()

    # Aggregate evaluation
    if y_pred_overall and y_true_overall:
        print("\nReporting global metrics ... ")
        # Concatenate aggregated data
        y_pred_overall = np.concatenate((y_pred_overall))
        y_true_overall = np.concatenate((y_true_overall))

        # Evaluate overall prediction against ground-truth
        evaluate(conf, y_true_overall, y_pred_overall, cf.id)





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
    fname = os.path.basename(config.output_path).replace('.', '_')

    mask_file = os.path.join(config.mask_path, fname + '.png')

    # Save output mask image to file (RGB -> BGR conversion)
    # Note that the default color format in OpenCV is often
    # referred to as RGB but it is actually BGR (the bytes are reversed).
    cv2.imwrite(mask_file, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

    print('Reconstructed mask saved to {}'.format(mask_file))


def get_output(conf, model, img_path, mask_path=None):
    """
    -------------------------------------
     Generate segmentation map for single input image
    -------------------------------------
     Inputs:     configuration settings     (dict)
                 model                      (Model)
                 image file path            (str)
     Outputs:    segmentation mask
    -------------------------------------
    """

    # Check if output exists already
    fname = os.path.basename(img_path).replace('.', '_')
    output_file = os.path.join(config.output_path, 'outputs', fname + '_output.pth')
    md_file = os.path.join(config.output_path, 'outputs', fname + '_md.json')
    mask_file = os.path.join(config.output_path, 'masks', fname + '.png')

    # Bypass model test and jump to evaluation of masks
    if os.path.exists(output_file) and input(
            "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(output_file)) != 'Y':
        return

    # Extract image subimages [NCWH format]
    img = utils.get_image(img_path, config.in_channels)
    w_full = img.shape[1]
    h_full = img.shape[0]

    print('\n---\nTest Image [{}]: {}'.format(fname, img_path))
    print('\tWidth: {}px'.format(w_full))
    print('\tHeight: {}px'.format(h_full))
    print('\tChannels: {}'.format(config.in_channels))

    if config.normalize_default:
        print('\tInput normalized to default mean: {}, std: {}'.format(params.px_mean_default,
                                                                       params.px_std_default))
    if config.resample:
        img = utils.get_image(config.img_path, config.in_channels, scale=config.resample)
        w_full = img.shape[1]
        h_full = img.shape[0]

        print('\n---\nResampled:')
        print('\tScaling: {}'.format(config.resample))
        print('\tWidth: {}px'.format(w_full))
        print('\tHeight: {}px'.format(h_full))

    # Set stride to half tile size
    stride = params.patch_size // 2

    # Adjust image size to fit N tiles
    img, w, h, offset = utils.adjust_to_tile(img, params.patch_size, stride, config.in_channels)
    print('\nImage Resized to: ')
    print('\tWidth: {}px'.format(w))
    print('\tHeight: {}px'.format(h))
    print('\tTop offset: {}'.format(offset))

    # Convert image to tensor
    img_data = torch.as_tensor(img, dtype=torch.float32)

    # Create image tiles
    print("\nCreating test image tiles ... ")
    img_data = img_data.unfold(0, params.patch_size, stride).unfold(1, params.patch_size, stride)
    img_data = torch.reshape(img_data,
                             (img_data.shape[0] * img_data.shape[1], config.in_channels, params.patch_size,
                              params.patch_size))
    n_samples = int(img_data.shape[0] * config.clip)

    print('\nImage Tiles: ')
    print('\tN: {}'.format(img_data.shape[0]))
    print('\tSize: {}px'.format(img_data.shape[2]))
    print('\tChannels: {}'.format(img_data.shape[1]))
    print('\tStride: {}'.format(stride))

    model.evaluator.metadata = {
        "w": w,
        "h": h,
        "w_full": w_full,
        "h_full": h_full,
        "offset": offset,
        "stride": stride,
        "n_samples": n_samples
    }

    print('\nProcessing image tiles ... ')
    with torch.no_grad():
        for i in trange(n_samples):
            x = img_data[i].unsqueeze(0).float()
            model.test(x)
            model.iter += 1

    # Save prediction test output to file
    model.evaluator.save(fname)
    print("Output data saved to {}.".format(model.evaluator.output_path))

    # Save full mask image to file
    if os.path.exists(mask_file) and input(
            "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(mask_file)) != 'Y':
        return

    y_pred = model.evaluator.save_image(fname)
    print("Output mask saved to {}.".format(model.evaluator.masks_path))


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