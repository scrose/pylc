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

import torch
from config import get_config
import utils.utils as utils
from models.base import Model
from tqdm import trange
from params import params


def test(conf, model):

    """ Test trained model """

    # Load test image
    img = utils.get_image(conf.img_path, conf.in_channels)
    w_full = img.shape[1]
    h_full = img.shape[0]

    print('\n---\nTest Image: {}'.format(conf.img_path))
    print('\tWidth: {}px'.format(w_full))
    print('\tHeight: {}px'.format(h_full))
    print('\tChannels: {}'.format(conf.in_channels))

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
                             (img_data.shape[0]*img_data.shape[1], conf.in_channels, params.patch_size, params.patch_size))
    print('\nImage Tiles: ')
    print('\tN: {}'.format(img_data.shape[0]))
    print('\tSize: {}px'.format(img_data.shape[2]))
    print('\tChannels: {}'.format(img_data.shape[1]))
    print('\tStride: {}'.format(stride))

    if conf.validate:
        # Load mask data subimages [NCWH]
        mask = utils.get_image(conf.mask_path, 3)

        print('\nMask (Validation): {}'.format(conf.img_path))
        print('\tWidth: {}px'.format(img_data.shape[1]))
        print('\tHeight: {}px'.format(img_data.shape[0]))
        print('\tChannels: {}'.format(conf.n_channels))

        # Adjust mask size to file N tiles
        mask, offset = utils.adjust_to_tile(mask, params.patch_size, stride, params.in_channels)
        print('\nMask resized and cropped to: ')
        print('\tWidth: {}px'.format(mask.shape[1]))
        print('\tHeight: {}px'.format(mask.shape[0]))
        print('\tTop clip size: {}px'.format(offset))

        # Convert mask to tensor and get tiles
        mask_data = torch.as_tensor(mask, dtype=torch.int64)
        mask_data = mask_data.unfold(0, params.patch_size, stride).unfold(1, params.patch_size, stride)
        mask_data = torch.reshape(mask_data,
                                  (mask_data.shape[0]*mask_data.shape[1], 3, params.patch_size, params.patch_size))
        print('\nMask Tiles: ')
        print('\tN: {}'.format(mask_data.shape[0]))
        print('\tSize: {}px x {}px'.format(mask_data.shape[1], mask_data.shape[2]))
        print('\tStride: {}px'.format(stride))

        # Merge dataset if palette mapping is provided
        print('\tConverting masks to class index encoding ... ', end='')
        if 'fortin' == conf.dset:
            # Encode masks to 1-hot encoding [NWH format] 11-class palette
            mask_data = utils.class_encode(mask_data, params.palette)
            print('done.')
            print('\tMerging current palette to alt palette ... ', end='')
            mask_data = utils.merge_classes(mask_data, params.categories_merged_alt)
            print('done.')
        else:
            # Encode masks to 1-hot encoding [NWH format] 9-class palette
            mask_data = utils.class_encode(mask_data, params.palette_alt)
            print('done.')

        print('\tMask Patches {} / Shape: {}'.format(mask_data.shape[0], mask_data.shape))
        assert img_data.shape[0] == mask_data.shape[0], 'Image dimensions must match mask dimensions.'

    n_samples = int(img_data.shape[0] * conf.clip)
    print('\nProcessing image tiles ... ')

    model.net.eval()
    with torch.no_grad():
        for i in trange(n_samples):
            x = img_data[i].unsqueeze(0).float()

            if conf.validate:
                y = mask_data[i].unsqueeze(0).long()
                model.eval(x, y, test=True)
                model.log()
            else:
                model.test(x)

            # Save model output to file
            if model.iter == int(conf.clip * n_samples):
                model.save(test=True)
                print("Output data saved to {}.".format(model.evaluator.output_file))
                exit()

            model.iter += 1

    if conf.validate:
        print("\n[Test] Dice Best: %4.4f\n" % model.loss.best_dice)

    # Save model output to file
    model.save(test=True)
    print("Output data saved to {}.".format(model.evaluator.output_file))

    # Save full mask image to file
    model.save_image(w, h, w_full, h_full, offset, stride)
    print("Output image saved to {}.".format(model.evaluator.img_file))
    exit()


def init_capture(conf):

    """ initialize parameters for capture type """
    if conf.capture == 'historic':
        conf.n_classes = 9
        conf.in_channels = 1
    elif conf.capture == 'repeat':
        conf.n_classes = 9
        conf.in_channels = 3
    return conf


def main(conf):

    # initialize conf parameters based on capture type
    conf = init_capture(conf)

    print("\n---\nBeginning test on {} model".format(conf.model))
    print("\tMode: {}".format(conf.mode))
    print("\tCapture Type: {}".format(conf.capture))
    print('\tModel Ref: {}\n\tDataset: {}'.format(conf.id, conf.dset))
    print('\tInput channels: {}\n\tClasses: {}'.format(conf.in_channels, conf.n_classes))
    print('\tTest Image: {}'.format(conf.img_path))
    print('\tTest Mask: {}'.format(conf.mask_path))

    model = Model(conf)
    print("\nPretrained model loaded.", end='')

    if conf.mode == params.NORMAL:
        test(conf, model)
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
