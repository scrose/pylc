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
import json
import os

# import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

import torch
from config import get_config
import utils.utils as utils
import utils.eval as metrics
from models.base import Model
from tqdm import trange
from params import params
import cv2
import numpy as np
import utils.tex as tex


def test(conf, model, bypass=False):
    """ Apply trained model to test dataset """

    # Initialize files list
    files = []
    y_true_overall = []
    y_pred_overall = []

    # iterate over available datasets
    for dset in params.dsets:

        # Check dataset configuration against parameters
        # COMBINED uses both dst-A, dst-B and dst-C
        print('\nLoading test images ... ')
        if params.COMBINED == conf.dset or dset == conf.dset:

            # get image/target file list
            img_dir = params.get_path('raw', dset, conf.capture, params.TEST, 'img')
            target_dir = params.get_path('raw', dset, conf.capture, params.TEST, 'mask')
            if not img_dir or not target_dir:
                print("\nNOTE: Skipping processing of dataset {} for {} pairs since not found.".format(dset, conf.capture))
                continue
            img_files = utils.load_files(img_dir, ['.tif', '.tiff', '.jpg', '.jpeg'])
            target_files = utils.load_files(target_dir, ['.png'])

            print('\tLooking for images in: {}'.format(img_dir))
            print('\tLooking for masks in: {}'.format(target_dir))

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
        y_pred = None

        # Check if output exists already
        fname = os.path.basename(img_path).replace('.', '_')
        output_file = os.path.join(config.output_path, 'outputs', fname + '_output.pth')
        md_file = os.path.join(config.output_path, 'outputs', fname + '_md.json')
        mask_file = os.path.join(config.output_path, 'masks', fname + '.png')

        # Bypass model test and jump to evaluation of masks
        if not bypass:
            get_output = True
            if os.path.exists(output_file) and input(
                    "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(output_file)) != 'Y':
                get_output = False
            if os.path.exists(mask_file) and input(
                    "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(mask_file)) != 'Y':
                get_output = False

            if get_output:
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

                # Create image tiles
                print("\nCreating test image tiles ... ")
                img_data = img_data.unfold(0, params.patch_size, stride).unfold(1, params.patch_size, stride)
                img_data = torch.reshape(img_data,
                                         (img_data.shape[0] * img_data.shape[1], conf.in_channels, params.patch_size,
                                          params.patch_size))
                n_samples = int(img_data.shape[0] * conf.clip)

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
                if conf.save_output:
                    model.evaluator.save(fname)
                    print("Output data saved to {}.".format(model.evaluator.output_path))

                # Save full mask image to file
                y_pred = model.evaluator.save_image(fname)
                print("Output mask saved to {}.".format(model.evaluator.masks_path))

        # Measure accuracy wrt. ground-truth
        if conf.validate:
            if not config.global_metrics and os.path.exists(md_file) and input(
                    "\tMetadata file {} exists. Re-do validation? (Type \'Y\' for yes): ".format(md_file)) != 'Y':
                continue
            # load ground-truth data
            print("\nStarting evaluation of outputs ... ")
            y_true = torch.as_tensor(utils.get_image(target_path, 3), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
            print("\tLoading mask file {}".format(mask_file))
            y_pred = torch.as_tensor(utils.get_image(mask_file, 3), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)

            # Class encode input predicted data
            y_pred = utils.class_encode(y_pred, params.palette_lcc_a)

            # Class encode target data
            if dset == 'dst-b':
                print('\t\tMapping LCC-B to LCC-A categories.'.format(conf.id))
                # Encode masks to 1-hot encoding [NWH format] 11-class LCC-B palette
                y_true = utils.class_encode(y_true, params.palette_lcc_b)
                y_true = utils.map_palette(y_true, params.lcc_btoa_key)
            # elif dset == 'dset-c':
            #     print('\t\tMapping LCC-C to LCC-A categories.'.format(conf.id))
            #     palette_key = params.lcc_ctoa_key
            #     # Encode masks to 1-hot encoding [NWH format] 11-class LCC-C palette
            #     y_true = utils.class_encode(y_true, params.palette_lcc_c)
            #     y_true = utils.map_palette(y_true, params.lcc_ctoa_key)
            else:
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
            if not config.global_metrics:
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
        evaluate(conf, y_true_overall, y_pred_overall, conf.id)


def evaluate(conf, y_true, y_pred, fid):
    """ Evaluate test output """

    dpi = 400
    font = {'weight': 'bold', 'size': 18}
    plt.rc('font', **font)
    sns.set(font_scale=0.9)
    labels = params.label_codes_dst_a

    # initialize metadata dict
    md = {
        'id': conf.id,
        'fid': fid
    }

    # # Ensure true mask has all of the categories
    target_idx = np.unique(y_true)
    input_idx = np.unique(y_pred)
    label_idx = np.unique(np.concatenate((target_idx, input_idx)))

    labels = []
    for idx in label_idx:
        labels += [params.label_codes_dst_a[idx]]

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    cmap_path = os.path.join(conf.output_path, 'outputs', fid + '_cmap.pdf')
    # np.save(os.path.join(conf.output_path, 'outputs', fname + '_cmap.npy'), conf_matrix)
    cmap = sns.heatmap(conf_matrix, vmin=0.01, vmax=1.0, fmt='.1g', xticklabels=labels, yticklabels=labels, annot=True)
    plt.ylabel('Ground-truth', fontsize=16, labelpad=6)
    plt.xlabel('Predicted', fontsize=16, labelpad=6)
    cmap.get_figure().savefig(cmap_path, format='pdf', dpi=dpi)
    plt.clf()

    # Ensure true mask has all of the categories
    for idx in range(len(params.label_codes_dst_a)):
        if idx not in target_idx:
            y_true[idx] = idx

    # Classification Report
    print(classification_report(y_true, y_pred, target_names=params.label_codes_dst_a, zero_division=0))
    md['report'] = classification_report(
        y_true, y_pred, target_names=params.label_codes_dst_a, output_dict=True, zero_division=0)

    # Weighted F1 Score (DSC)
    md['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print('Weighted F1 Score: {}'.format(md['f1']))

    # Weighted Jaccard (ioU)
    md['iou'] = jaccard_score(y_true, y_pred, average='weighted')
    print('Weighted IoU: {}'.format(md['iou']))

    # Matthews correlation coefficient
    md['mcc'] = matthews_corrcoef(y_true, y_pred)
    print('MCC: {}'.format(md['mcc']))

    # Save image metadata
    with open(os.path.join(conf.output_path, 'outputs', fid + '_md.json'), 'w') as fp:
        json.dump(md, fp, indent=4)

    # Save metadata as latex table
    # write back the new document
    with open(os.path.join(conf.output_path, 'outputs', fid + '_md.tex'), 'w') as fp:
        md_tex = tex.convert_md_to_tex(md)
        fp.write(md_tex)


def reconstruct(conf):
    """ Reconstruct mask from unary output """

    # load output data
    output = torch.load(conf.output_path, map_location=lambda storage, loc: storage)
    print('Loaded results for {}'.format(conf.output_path))

    # get unary output data / metadata
    if type(output['results']) == list:
        unary_data = np.concatenate(output['results'])
    else:
        unary_data = output['results'].numpy()

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


def get_output(conf, model, img_path, target_path=None):

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
    img = utils.get_image(img_path, conf.in_channels)
    w_full = img.shape[1]
    h_full = img.shape[0]

    print('\n---\nTest Image [{}]: {}'.format(fname, img_path))
    print('\tWidth: {}px'.format(w_full))
    print('\tHeight: {}px'.format(h_full))
    print('\tChannels: {}'.format(conf.in_channels))

    if conf.normalize_default:
        print('\tInput normalized to default mean: {}, std: {}'.format(params.px_mean_default,
                                                                       params.px_std_default))
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

    # Create image tiles
    print("\nCreating test image tiles ... ")
    img_data = img_data.unfold(0, params.patch_size, stride).unfold(1, params.patch_size, stride)
    img_data = torch.reshape(img_data,
                             (img_data.shape[0] * img_data.shape[1], conf.in_channels, params.patch_size,
                              params.patch_size))
    n_samples = int(img_data.shape[0] * conf.clip)

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

    # Load model for testing or evaluation
    if conf.mode == params.NORMAL or conf.mode == params.EVALUATE or conf.mode == params.SINGLE:
        model = Model(conf)
        print("\n---\nBeginning test on {} model".format(conf.model))
        print("\tMode: {}".format(conf.mode))
        print("\tCapture Type: {}".format(conf.capture))
        print('\tModel Ref: {}\n\tDataset: {}'.format(conf.id, conf.dset))
        print('\tInput channels: {}\n\tClasses: {}'.format(conf.in_channels, conf.n_classes))
        print('\tDataset: {}'.format(conf.dset))
        print("\nPretrained model loaded.")

        if conf.mode == params.EVALUATE:
            print("\nEvaluation of model {} ... ".format(conf.id))
            test(conf, model, bypass=True)
        elif conf.mode == params.SINGLE:
            print("\nTest single image on model {} ... ".format(conf.id))
            get_output(conf, model, conf.img_path, conf.mask_path)
        else:
            print("\nTesting of model {} ... ".format(conf.id))
            test(conf, model)

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
