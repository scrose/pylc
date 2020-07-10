"""
    ------------------------
    Mountain Legacy Project: Semantic Segmentation of Oblique Landscape Photographs
    Author: Spencer Rose
    Date: May 2020
    University of Victoria

    REFERENCES:
    ------------------------
    Long, Jonathan, Evan Shelhamer, and Trevor Darrell.
    "Fully convolutional networks for semantic segmentation." In Proceedings of
    the IEEE conference on computer vision and pattern recognition, pp. 3431-3440.
    2015  https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    ------------------------
    U-Net
    ------------------------
    DeepLab
"""

import os
import torch
from config import get_config
import utils.utils as utils
import utils.eval as metrics
from models.base import Model
from tqdm import trange
from params import params
import cv2
import numpy as np


def test(conf, model):
    """ Apply trained model to test dataset """

    # Initialize files list
    files = []

    # iterate over available datasets
    for dset in params.dsets:

        # Check dataset configuration against parameters
        # COMBINED uses both dst-A, dst-B and dst-C
        if params.COMBINED == conf.dset or dset == conf.dset:

            # get image/target file list
            img_dir = params.get_path('raw', dset, conf.capture, params.TEST, 'img')
            target_dir = params.get_path('raw', dset, conf.capture, params.TEST, 'mask')
            img_files = utils.load_files(img_dir, ['.tif', '.tiff', '.jpg', '.jpeg'])
            target_files = utils.load_files(target_dir, ['.png'])

            print('\tLooking at images in: {}'.format(img_dir))
            print('\tLooking at maskes in: {}'.format(target_dir))

            # verify image/target pairing
            f_idx = 0
            for f_idx, img_fname in enumerate(img_files):
                assert f_idx < len(target_files), 'Image {} does not have a target.'.format(img_fname)
                target_fname = target_files[f_idx]
                assert os.path.splitext(img_fname)[0] == os.path.splitext(target_fname)[0].replace('_mask', ''), \
                    'Image {} does not match target {}.'.format(img_fname, target_fname)

                # re-add full path to image and associated target data
                img_fname = os.path.join(img_dir, img_fname)
                target_fname = os.path.join(target_dir, target_fname)
                files += [{'img': img_fname, 'mask': target_fname, 'dset': dset}]

            # Validate image-target correspondence
            assert f_idx < len(target_files), 'target {} does not have an image.'.format(target_files[f_idx])

    # Extract image/target subimages and save to database
    print("\n{} image/target pairs found.".format(len(files)))

    # Run test on input images
    print('\nRunning model test ... ')
    for f_idx, fpair in enumerate(files):
        # Get image and associated target data
        img_path = fpair.get('img')
        target_path = fpair.get('mask')
        dset = fpair.get('dset')

        # Check if output exists already
        fname = os.path.basename(img_path).replace('.', '_')
        output_file = os.path.join(config.output_path, 'outputs', fname + '_output.pth')
        mask_file = os.path.join(config.output_path, 'masks', fname + '.png')
        if os.path.exists(output_file) and input(
                "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(output_file)) != 'Y':
            continue
        if os.path.exists(mask_file) and input(
                "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(mask_file)) != 'Y':
            continue

        # Extract image subimages [NCWH format]
        img = utils.get_image(img_path, conf.in_channels)
        w_full = img.shape[1]
        h_full = img.shape[0]

        print('\n---\nTest Image [{}]: {}'.format(f_idx, img_path))
        print('\tWidth: {}px'.format(w_full))
        print('\tHeight: {}px'.format(h_full))
        print('\tChannels: {}'.format(conf.in_channels))

        if conf.normalize_default:
            print('\tInput normalized to default mean: {}, std: {}'.format(params.px_mean_default, params.px_std_default))

        if conf.resample:
            img = utils.get_image(conf.img_path, conf.in_channels, scale=conf.resample)
            w_full = img.shape[1]
            h_full = img.shape[0]

            print('\n---\nResampled:')
            print('\tScaling: {}'.format(conf.resample))
            print('\tWidth: {}px'.format(w_full))
            print('\tHeight: {}px'.format(h_full))

        # Set stride to half tile size
        stride = params.patch_size // 2

        # Adjust image size to fit N tiles
        img, w, h, offset = utils.adjust_to_tile(img, params.patch_size, stride, conf.in_channels)
        print('\nImage Resized to: ')
        print('\tWidth: {}px'.format(w))
        print('\tHeight: {}px'.format(h))
        print('\tTop offset: {}'.format(offset))

        # Convert image to tensor
        img_data = torch.as_tensor(img, dtype=torch.float32)
        mask_data = None

        # Create image tiles
        print("\nCreating test image tiles ... ")
        img_data = img_data.unfold(0, params.patch_size, stride).unfold(1, params.patch_size, stride)
        img_data = torch.reshape(img_data,
                                 (img_data.shape[0] * img_data.shape[1], conf.in_channels, params.patch_size,
                                  params.patch_size))
        print('\nImage Tiles: ')
        print('\tN: {}'.format(img_data.shape[0]))
        print('\tSize: {}px'.format(img_data.shape[2]))
        print('\tChannels: {}'.format(img_data.shape[1]))
        print('\tStride: {}'.format(stride))

        conf.validate = False

        # if conf.validate:
        #     # Load mask data subimages [NCWH]
        #     mask = utils.get_image(conf.mask_path, 3)
        #
        #     print('\nMask (Validation): {}'.format(conf.mask_path))
        #     print('\tWidth: {}px'.format(mask.shape[1]))
        #     print('\tHeight: {}px'.format(mask.shape[0]))
        #     print('\tChannels: {}'.format(conf.n_channels))
        #
        #     # Adjust mask size to fit N tiles
        #     mask, w_mask, h_mask, offset = utils.adjust_to_tile(mask, params.patch_size, stride, conf.in_channels,
        #                                                         interpolate=cv2.INTER_NEAREST)
        #     assert w == w_mask and h == h_mask, print("Mask resized dimensions {}x{} must match image dimensions {}x{}.".format(w, h, w_mask, h_mask))
        #     print('\nMask resized and cropped to: ')
        #     print('\tWidth: {}px'.format(mask.shape[1]))
        #     print('\tHeight: {}px'.format(mask.shape[0]))
        #     print('\tTop clip size: {}px'.format(offset))
        #
        #     # Convert mask to tensor and get tiles
        #     mask_data = torch.as_tensor(mask, dtype=torch.int64)
        #     mask_data = mask_data.unfold(0, params.patch_size, stride).unfold(1, params.patch_size, stride)
        #     mask_data = torch.reshape(mask_data,
        #                               (mask_data.shape[0]*mask_data.shape[1], 3, params.patch_size, params.patch_size))
        #     print('\nMask Tiles: ')
        #     print('\tN: {}'.format(mask_data.shape[0]))
        #     print('\tSize: {}px'.format(mask_data.shape[2]))
        #     print('\tChannels: {}'.format(mask_data.shape[1]))
        #     print('\tStride: {}'.format(stride))
        #
        #     # Merge dataset if palette mapping is provided
        #     print('\tConverting masks to class index encoding ... ', end='')
        #     if 'fortin' == conf.dset:
        #         # Encode masks to 1-hot encoding [NWH format] 11-class palette
        #         mask_data = utils.class_encode(mask_data, params.palette_lcc_b)
        #         print('done.')
        #         print('\tMerging current palette to alt palette ... ', end='')
        #         mask_data = utils.merge_classes(mask_data, params.categories_merged_lcc_a)
        #         print('done.')
        #     else:
        #         # Encode masks to 1-hot encoding [NWH format] 9-class palette
        #         mask_data = utils.class_encode(mask_data, params.palette_lcc_a)
        #         print('done.')
        #
        #     print('\tMask Patches {} / Shape: {}'.format(mask_data.shape[0], mask_data.shape))
        #     assert img_data.shape[0] == mask_data.shape[0], 'Image dimensions must match mask dimensions.'

        n_samples = int(img_data.shape[0] * conf.clip)

        model.evaluator.metadata = {
            "w": w,
            "h": h,
            "w_full": w_full,
            "h_full": h_full,
            "offset": offset,
            "stride": stride,
            "n_samples": n_samples
        }

        model.net.eval()

        print('\nProcessing image tiles ... ')
        with torch.no_grad():
            for i in trange(n_samples):
                x = img_data[i].unsqueeze(0).float()

                if conf.validate:
                    y = mask_data[i].unsqueeze(0).long()
                    model.eval(x, y, test=True)
                    model.log()
                else:
                    model.test(x)

                model.iter += 1

        if conf.validate:
            print("\n[Test] Dice Best: %4.4f\n" % model.loss.best_dice)

        # Save prediction test output to file
        model.evaluator.save(fname)
        print("Output data saved to {}.".format(model.evaluator.output_path))

        # Save full mask image to file
        model.evaluator.save_image(fname)
        print("Output mask saved to {}.".format(model.evaluator.masks_path))

        # Reset evaluator
        model.evaluator.reset()


def eval(conf):
    """ Evaluate test output """

    # load output data
    output = np.concatenate(torch.load(conf.output_path, map_location=lambda storage, loc: storage)['results'])
    print('Loaded results for {}'.format(conf.output_path))
    output = torch.tensor(output).float()

    eval_dir = '/Users/boutrous/Workspace/MLP/experiments/DLB-H-4.1/'

    mask_data = utils.get_image(config.mask_path)

    # load ground-truth target

    # Evaluate metrics
    print('\nMetrics:')

    # DSC: Raw unary outputs
    # dsc1 = metrics.dsc(output, mask_data.long())
    # print('\nDSC of raw unary values:'.format(dsc1))

    # DSC: Mapped unary
    output_map = torch.argmax(output, dim=1)

    # One-hot encoding
    y_true_1hot = torch.nn.functional.one_hot(mask_data.long(), num_classes=params.n_classes).permute(0, 3, 1, 2)
    output_1hot = torch.nn.functional.one_hot(output_map.long(), num_classes=params.n_classes).permute(0, 3, 1, 2)

    # Save mapped input/target for further analysis
    np.save(os.path.join(eval_dir, 'target_' + conf.id), mask_data)
    np.save(os.path.join(eval_dir, 'input_' + conf.id), output_map)

    # Calculate true postives
    tp = torch.sum(output_1hot * y_true_1hot, dim=(0, 2, 3))
    # Calculate false positives
    fp = np.sum(output_1hot * (1 - y_true_1hot), axis=0)

    # compute mean of y_true U y_pred / (y_pred + y_true)
    # cardinality = torch.sum(output_1hot + y_true_1hot, dim=(0, 2, 3))
    # dsc2 = (2. * intersection + params.dice_smooth) / (cardinality + params.dice_smooth)

    # print(dsc2)
    # print(dsc2.mean().item())

    px_dist_ytrue = np.sum(y_true_1hot.numpy(), axis=(0, 2, 3))
    px_dist_output = np.sum(output_1hot.numpy(), axis=(0, 2, 3))
    px_count = np.sum(px_dist_ytrue)

    fiou = 0.
    fw = px_dist_ytrue / px_count

    print('\tTotal pixel count: {}'.format(px_count))
    print('\tPixel Dist (Ground-truth: {}'.format(px_dist_ytrue))
    print('\tPixel Dist (Output): {}'.format(px_dist_output))
    print('\tTrue Positives: {}'.format(tp.numpy()))
    print('\tFalse Positives: {}'.format(fp))

    for i in range(params.n_classes):
        fiou += (px_dist_ytrue[i] * tp) / (px_dist_ytrue[i] + np.sum(fp[i]))

    fiou /= px_count
    print(fiou)


def reconstruct(conf):
    """ Reconstruct mask from unary output """

    # load output data
    output = torch.load(conf.output_path, map_location=lambda storage, loc: storage)
    print('Loaded results for {}'.format(conf.output_path))

    # get unary output data / metadata
    unary_data = np.concatenate(output['results'])
    md = output['metadata']

    # Reconstruct seg-mask from predicted tiles
    mask_img = utils.reconstruct(unary_data, md)

    # Extract image path
    fname = os.path.basename(conf.output_path).replace('.', '_')

    mask_file = os.path.join(conf.mask_path, fname + '.png')

    # Save output mask image to file (RGB -> BGR conversion)
    # Note that the default color format in OpenCV is often
    # referred to as RGB but it is actually BGR (the bytes are reversed).
    cv2.imwrite(mask_file, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

    print('Reconstructed mask saved to {}'.format(mask_file))


def init_capture(conf):
    """ force initialize parameters for capture type """
    if conf.capture == 'historic':
        conf.n_classes = params.n_classes
        conf.in_channels = 1
    elif conf.capture == 'repeat':
        conf.n_classes = params.n_classes
        conf.in_channels = 3
    return conf


def main(conf):
    # initialize conf parameters based on capture type
    conf = init_capture(conf)

    # Run model on test image
    if conf.mode == params.NORMAL:
        model = Model(conf)
        print("\n---\nBeginning test on {} model".format(conf.model))
        print("\tMode: {}".format(conf.mode))
        print("\tCapture Type: {}".format(conf.capture))
        print('\tModel Ref: {}\n\tDataset: {}'.format(conf.id, conf.dset))
        print('\tInput channels: {}\n\tClasses: {}'.format(conf.in_channels, conf.n_classes))
        print('\tDataset: {}'.format(conf.dset))
        print("\nPretrained model loaded.", end='')
        test(conf, model)
    # Evaluate test outputs
    elif conf.mode == params.EVALUATE:
        print("\nEvaluation of output unary data started ... ")
        eval(conf)
    # Reconstruct mask outputs
    elif conf.mode == params.RECONSTRUCT:
        print("\nReconstruct mask from output unary data ... ")
        reconstruct(conf)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(conf.mode))


if __name__ == "__main__":

    """ Parse model confuration """
    config, unparsed, parser = get_config(params.TEST)

    # If we have unparsed arguments, or help request print usage and exit
    if len(unparsed) > 0 or config.h:
        parser.print_usage()
        exit()

    main(config)

#
# test.py ends here
