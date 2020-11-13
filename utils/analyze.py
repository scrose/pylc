"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
An evaluation of deep learning semantic segmentation for
land cover classification of oblique ground-based photography
MSc. Thesis 2020.
<http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Analyzer Utilities
File: analyze.py
"""

import os, sys, glob, math, json
import numpy as np
import torch
import h5py
import random
from random import shuffle
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter    
from config import defaults

# paths 
root = './'

# ===========================
# File Utilities
# ===========================

# Split paths into directory list
def split_path(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

# Load image pairs by extension
def load_files(path, exts):
    path = os.path.join(root, path)
    assert os.path.exists(path), 'Directory path {} does not exist.'.format(path)
    return list(sorted([os.path.join(path, f) for f in os.listdir(path) if any(ext in f for ext in exts)]))


# load data from database
def load_db(db_path, offset=0):
    # Load database dataset
    assert os.path.exists(db_path)==True, 'Database path {} does not exist'.format(db_path)
    print("Loading dataset from {} ... ".format(db_path), end='')
    # Load hdf5 file
    fdata = h5py.File(db_path, mode='r')
    # Load data
    img_data = fdata.get('img')[()]
    mask_data = fdata.get('mask')[()]
    print('done.')
    # Close database file
    fdata.close()
    return img_data, mask_data


# load trained model
def load_models(exps, from_checkpoint=False):

    models = []
    losses = []
    dirs = []

    # load models and loss data
    for exp in exps:

        # get experiment directory path
        edir = os.path.join(exp_dir, exp)
        if not os.path.exists(edir):
            print("Error: experiment path {} was not found.".format(edir))
        else:
            dirs += [edir]

        # get file paths
        model_file = os.path.join(edir, "model.pth")
        losses_file = os.path.join(edir, "losses.pth")

        # Losses file
        if not os.path.exists(losses_file):
            print('Losses file {} not found.'.format(losses_file))
        else:
            losses += [torch.load(losses_file, map_location=torch.device('cpu'))]
            print('Loaded losses for {}'.format(losses_file))

        # Load model from checkpoint
        if from_checkpoint:
            checkpoint_file = os.path.join(edir, "checkpoint.pth")
            if not os.path.exists(checkpoint_file):
                print('Checkpoint file {} not found.'.format(checkpoint_file))
            else:
                print('Loading model from checkpoint file {} ...'.format(checkpoint_file))
                checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
                model = checkpoint["model"]
                optim = checkpoint["optim"]
                torch.save({
                    "model": model,
                    "optim": optim,
                }, model_file)

        # Load model
        if not os.path.exists(model_file):
            print('Model file {} not found.'.format(model_file))
        else:
            models += [torch.load(model_file, map_location=torch.device('cpu'))]
            print('Loaded model for {}'.format(model_file))

    return models, losses, dirs


# load trained model
def load_model(model_path, from_checkpoint=False):

    model = None

    # check model path
    if not os.path.exists(model_path):
        print("Error: model path was not found:\n\t{}.".format(model_path))

    # Load model from checkpoint
    if from_checkpoint:
        print('Loading model from checkpoint file:\t\n{}'.format(model_path))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = checkpoint["model"]
        optim = checkpoint["optim"]

    # Load model directly
    else:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print('Loaded model at path:\n\t{}'.format(model_path))

    return model


# load test output data from model
def load_output(output_file):

    # Output results
    if not os.path.exists(output_file):
        print('Output file {} not found.'.format(output_file))
        return
    else:
        output = torch.load(output_file, map_location=lambda storage, loc: storage)
        print('Loaded results for {}'.format(output_file))
        return output


# load multiple test outputs data from model
def load_outputs(exps, filename):

    output = []

    # load models and loss data
    for exp in exps:

        edir = os.path.join(exp_dir, exp)
        output_file = os.path.join(edir, filename + ".pth")

        # Output results
        if not os.path.exists(output_file):
            print('Output file {} not found.'.format(output_file))
        else:
            output += [torch.load(output_file, map_location=lambda storage, loc: storage)]
            print('Loaded results for {}'.format(output_file))

    return output

# open dataset file pointer
def open_db(path):
    # Reference hdf5 db file
    return h5py.File(add_root(path), mode='r')


# get size of dataset
def size_db(db):
    return len(db['img'])

# add root path
def add_root(path):
    return os.path.join(root, os.path.normpath(path))


# Retrieves dataset metrics and statistics from file
# Input: path to metadata file
# Output: sample rates, other metrics/analytics
def load_data(path):
    # Load metadata
    metadata = np.load(add_root(path), allow_pickle=True)
    px_dist = metadata.item().get('px_dist')
    px_count = metadata.item().get('px_count')
    dset_px_dist = metadata.item().get('dset_px_dist')
    dset_px_count = metadata.item().get('dset_px_count')
    weights = metadata.item().get('weights')

    # Calculate class probabilities
    dset_probs = dset_px_dist/dset_px_count
    n_samples = len(px_dist)

    return px_dist, px_count, dset_px_dist, dset_px_count, dset_probs, weights


# Retrieves dataset metrics and statistics from file
# Input: path to metadata file
# Output: sample rates, other metrics/analytics
def save_data(path, metadata):
    np.save(add_root(path), metadata)
    print('Data saved to: {}'.format(path))


# ===========================
# Image Processing: Utilities
# ===========================


# Read image and reverse channel order
# Loads image as 8 bit (regardless of original depth)
def get_image(image_path, img_ch=3):
    assert img_ch == 3 or img_ch == 1, 'Invalid input channel number.'
    image = None
    if img_ch == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif img_ch == 3:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Replace class index values with palette colours
# Input format: NWH (class encoded)
def colourize(img_data, palette=None):
    
    if palette is None:
        palette = defaults.palette_rgb

    n = img_data.shape[0]
    w = img_data.shape[1]
    h = img_data.shape[2]

    # make 3-channel (RGB) image
    img_data = np.moveaxis(np.stack((img_data,)*3, axis=1), 1, -1).reshape(n*w*h, 3)

    # map categories to palette colours
    for i, colour in enumerate(palette):
        bool = img_data == np.array([i,i,i])
        bool = np.all(bool, axis=1)
        img_data[bool] = palette[i]
    return img_data.reshape(n, w, h, 3)


# -----------------------------------
# Merge segmentation classes
# -----------------------------------
def merge_classes(data_tensor, merged_classes):
    data = data_tensor.numpy()

    # merge classes
    for i, cat_grp in enumerate(merged_classes):
        data[np.isin(data, cat_grp)] = i

    return torch.tensor(data)


# Convert RBG mask array to class-index encoded values
# input form: NCWH with RGB-value encoding, where C = RGB (3)
# Palette paramters in form [CC'], where C = number of classes, C' = 3 (RGB)
# output form: NCWH with one-hot encoded classes, where C = number of classes
def class_encode(input_data, palette=defaults.palette_rgb):
    assert input_data.shape[1] == 3, "Input data must be 3 channel (RGB)"
    input_data = input_data.to(torch.float32).mean(dim=1)
    palette = torch.from_numpy(palette).to(torch.float32).mean(dim=1)
    # map mask colours to segmentation classes
    for idx, c in enumerate(palette):
        bool = input_data == c
        input_data[bool] = idx
    return input_data.to(torch.uint8)


# -----------------------------------
# Apply affine distortion to image
# Input image [NWHC] / Mask [NWHC]
# Output image [NWHC] / Mask [NWHC]
# -----------------------------------
def augment_transform(img, mask, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    dim = img.shape[2]
    n_ch = img.shape[1]
    img = np.squeeze(np.moveaxis(img, 1, -1), axis=0)
    mask = np.squeeze(mask, axis=0)

    # apply random vertical flip
    if bool(random.getrandbits(1)):
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)

    # # Random affine deformation
    # center_square = np.float32(dsize) // 2
    # square_size = min(dsize) // 3
    #     pts1 = np.float32(
    #         [center_square + square_size, [center_square[0] + square_size, center_square[1]
    #                                        - square_size], center_square - square_size])
    #     pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    #     M = cv2.getAffineTransform(pts1, pts2)
    #     image = cv2.warpAffine(image, M, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    #     mask = cv2.warpAffine(mask, M, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    # Apply random perspective shift
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    m_trans = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, m_trans, (dim, dim), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpPerspective(mask, m_trans, (dim, dim), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    if n_ch == 3:
        img = np.moveaxis(img, -1, 0)

    return img, mask


# Parameter Grid Search: Data augmentation
def grid_search(px_dist, px_count, dset_px_dist, dset_px_count, dset_probs, n_classes, rate_max):
    profile_data = []
    eps = 1e-8

    # Initialize oversample filter and class prior probabilities
    oversample_filter = np.clip(1/n_classes - dset_probs, a_min=0, a_max=1.)
    probs = px_dist/px_count
    probs_weighted = np.multiply(np.multiply(probs, 1/dset_probs), oversample_filter)
    scores = np.sqrt(np.sum(probs_weighted, axis=1))

    # Initialize augmentation parameters
    # rate coefficient ranges from 1 - 21
    rate_coefs = np.arange(1, 21, 1)
    # threshold between 
    thresholds = np.arange(0, 3., 0.05)
    aug_n_samples_max = 4000
    #delta_px = []
    jsd = []

    # initialize balanced model distribution
    balanced_px_prob = np.empty(n_classes)
    balanced_px_prob.fill(1/n_classes)

    # Grid search
    for i, rate_coef, in enumerate(rate_coefs):
        for j, threshold in enumerate(thresholds):
            assert rate_coef >= 1, 'Rate coefficient must be >= one.'
            oversample = scores > threshold
            rates = np.multiply(oversample, rate_coef * scores).astype(int)
            # clip rates to max value 
            rates = np.clip(rates, 0, rate_max)
            # limit to max number of augmented images
            if np.sum(rates) < aug_n_samples_max:
                aug_px_dist = np.multiply(np.expand_dims(rates, axis=1), px_dist)
                full_px_dist = px_dist + aug_px_dist
                full_px_probs = np.sum(full_px_dist, axis=0)/np.sum(full_px_dist)
                #delta_px += [np.sum(np.multiply(full_px_probs, oversample_filter))]
                jsd_sample = JSD(full_px_probs, balanced_px_prob)
                jsd += [jsd_sample]
                profile_data += [{
                    'probs': full_px_probs,
                    'threshold' : threshold,
                    'rate_coef': rate_coef,
                    'rates': rates,
                    'n_samples': int(np.sum(full_px_dist)/px_count),
                    'aug_n_samples': np.sum(rates),
                    'rate_max': rate_max,
                    'jsd':jsd_sample
                }]

    # Get parameters that minimize Jensen-Shannon Divergence metric
    if len(jsd) == 0:
        print('No augmentation optimization found')
        return
    optim_idx = np.argmin(np.asarray(jsd))
    return profile_data[optim_idx]



# Print colour-coded categories
def plot_legend(palette, title, categories=None):
    
    if categories is None:
        categories = defaults.class_labels_hex

    cell_width = 260
    cell_height = 30
    swatch_width = 48
    margin = 12
    topmargin = 40

    n = len(palette)
    ncols = 3
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, colour in enumerate(palette):
        row = i % nrows
        col = i // nrows
        y = row * cell_height
        label = ''

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, categories[colour] + label, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=colour, linewidth=18)

    plt.show()



# Plot image/mask samples
def plot_samples(img_data, mask_data, n_rows=20, n_cols=5, offset=0, title=None, palette=None, categories=None):
    
    if categories is None:
        categories = defaults.class_labels_hex

    if palette is None:
        palette = defaults.palette_rgb
        
    plot_legend(palette, title, categories=categories)

    k = offset
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex='col',
        sharey='row',
        figsize=(40, 100),
        subplot_kw={'xticks': [], 'yticks': []})
    plt.rcParams.update({'font.size': 18})

    # Plot sample subimages
    for i in range(0, n_rows):
        for j in range(n_cols):
            # Get sample patch & mask
            img = img_data[k].astype(int)
            mask = mask_data[k]

            # show original subimage (greyscale: number of dimensions = 3)
            if img_data.ndim == 3:
                axes[i,j].imshow(img, cmap='gray')
            else:
                axes[i,j].imshow(img)
            # overlay subimage seg-mask
            axes[i,j].imshow(mask, alpha=0.5)
            axes[i,j].set_title('Sample #{}'.format(k + 1))
            k += 1
    plt.show()


# Plot dataset pixel distribution profile
def plot_profile(
        n_classes,
        probs,
        n_samples,
        category_labels=defaults.class_labels,
        palette=defaults.palette_hex,
        title=None,
        overlay=None,
        label_1=None,
        label_2=None,
        save_path=None):

    # Calculate JSD and weights
    balanced_px_prob = np.empty(n_classes)
    balanced_px_prob.fill(1/n_classes)
    jsd = JSD(probs, balanced_px_prob)
    weights = eval_weights(probs)

    print(title)
    print('\n{:20s}{:>15s}{:>15s}\n'.format('Class', 'Prob', 'Weight', ))
    for i, p in enumerate(probs):
        print('{:20s}{:15f}{:15f}'.format(category_labels_dst_b[i], p, weights[i]))
    print('\nTotal samples: {}'.format(n_samples))
    print('Sample Size: {} x {} = {} pixels'.format(patch_size, patch_size, patch_size*patch_size))
    print('\nJSD: {:6f}'.format(jsd))

    # Set figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 18
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams.update({'font.size': 9})

    x = np.arange(n_classes)
    if overlay is not None:
        width = 0.3  # the width of the bars
        plt.bar(x - width/2, overlay, width=width, color=palette, alpha=0.5, label=label_2)
        plt.bar(x + width/2, probs, width=width, color=palette, alpha=0.9, label=label_1)
        plt.legend()
    else:
        plt.bar(x, probs, width=0.7, color=palette, alpha=0.9)
    plt.xticks(x, category_labels_dst_b)
    plt.ylabel('Proportion of pixels')
    plt.title(title)
    plt.axhline(1/n_classes, color='k', linestyle='dashed', linewidth=1)

    # Save plots
    if save_path:
        plt.savefig(os.path.join(save_path, str('pixel_dist.pdf')), dpi=200)

    plt.show()



# Plot sample pixel distribution profile
def plot_grid_profiles(probs, profile_data, n_rows=25, n_cols=5, offset=0):

    threshold = profile_data['threshold']
    rate_coef = profile_data['rate_coef']
    rates = profile_data['rates']
    jsd = profile_data['jsd']
    k = offset
    width = 0.5  # the width of the bars
    fig, axes = plt.subplots(n_rows, n_cols, sharex='col', sharey='row', figsize=(40, 300),
                             subplot_kw={'xticks': [], 'yticks': []})
    plt.rcParams.update({'font.size': 14})

    print('Threshold: {} \ Rate coefficient: {}'.format(threshold, rate_coef))

    # Plot sample class profiles
    for i in range(0, n_rows):
        for j in range(n_cols):
            # plot profile histogram
            rate = rates[k]
            title = 'Sample #{} [JSD: {} \ Rate: {}]'.format(k, jsd, rate)
            axes[i,j].axhline(1/n_classes, color='k', linestyle='dashed', linewidth=1)
            axes[i,j].bar(
                range(n_classes),
                probs[k],
                color=defaults.palette_hex,
                alpha=0.7
            )
            axes[i,j].set_title(title)
            k += 1
    plt.show()




# Plot loss curves
def plot_losses(data, title, fname, save_path=None, data_format='simple', focal=False, lr=True):

    # Format: [(x_0, ce_0, dsc_0)...]
    if data_format == 'simple':
        # Format: [(x_0, (ce_0, dsc_0, fl_0))...] 
        if focal:
            x_tr, ce_tr, dice_tr, focal_tr = zip(*data['train'])
            x_va, ce_va, dice_va, focal_va = zip(*data['valid'])
        # Format: [(x_0, ce_0, dsc_0)...]
        else:
            x_tr, ce_tr, dice_tr = zip(*data['train'])
            x_va, ce_va, dice_va = zip(*data['valid'])
    # Format: [(x_0, (ce_0, dsc_0))...] 
    elif data_format == 'nested':
        x_tr, loss_tr = zip(*data['train'])
        ce_tr, dice_tr = zip(*loss_tr)
        x_va, loss_va = zip(*data['valid'])
        ce_va, dice_va = zip(*loss_va)

    else:
        print('Data format {} not found.'.format(data_format))
        return

    plt.figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')

    # Plot losses
    plt.rcParams.update({'font.size': 14})
    plt.subplot(3, 1, 1)
    plt.title(title)
    plt.plot(x_tr, ce_tr, 'lightblue')
    yhat = savgol_filter(ce_tr, 51, 3) # window size 51, polynomial order 3
    plt.plot(x_tr, yhat, color='navy')
    plt.plot(x_va, ce_va, color='orange')
    plt.legend(['train', 'filtered','valid'])
    #for xc in x_lr:
    #plt.axvline(x=xc, color='pink')
    plt.ylabel('CE Loss')

    plt.subplot(3, 1, 2)
    dice_tr = [d_loss for d_loss in dice_tr]
    dice_va = [d_loss for d_loss in dice_va]
    # invert dice loss to mIoU
    plt.plot(x_tr, dice_tr, 'lightblue')
    yhat = savgol_filter(dice_tr, 51, 3) # window size 51, polynomial order 3
    plt.plot(x_tr, yhat, color='navy')
    plt.plot(x_va, dice_va, 'orange')
    plt.legend(['train','filtered','valid'])
    plt.xlabel('Iterations')
    plt.ylabel('DSC Loss')

    # Plot Focal loss (optional)
    if focal:
        plt.subplot(3, 1, 3)
        focal_tr = [f_loss for f_loss in focal_tr]
        focal_va = [f_loss for f_loss in focal_va]
        # invert dice loss to mIoU
        plt.plot(x_tr, focal_tr, 'lightblue')
        yhat = savgol_filter(focal_tr, 51, 3) # window size 51, polynomial order 3
        plt.plot(x_tr, yhat, color='navy')
        plt.plot(x_va, focal_va, 'orange')
        plt.legend(['train','filtered','valid'])
        plt.xlabel('Iterations')
        plt.ylabel('Focal Loss')
    
    # Plot learning rate steps
    if lr:
        x_lr, lr_tr = zip(*data["lr"])
        plt.subplot(3, 1, 4)
        plt.plot(x_lr, lr_tr, 'red')
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.ylabel('Learning Rate')

    # Save plots
    if save_path:
        plt.savefig(os.path.join(save_path, str(fname + '.pdf')), dpi=400)



def plot_eval(grid_data, mode, title, fname, save_path=None):

    # configure plott
    #plt.figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.title(title)
    plt.rcParams.update({'font.size': 14})
    markers = ["s", "o", "v"]


    # CE Loss
    plt.subplot(3, 1, 1)
    plt.ylabel('CE Loss')
    plt.xlabel('Iterations')
    #plt.legend(['train', 'filtered','valid'])
    for i, data in enumerate(grid_data):
        x_tr, ce_tr, dice_tr = zip(*data['train'])
        x_va, ce_va, dice_va = zip(*data['valid'])
        x_lr, lr_tr = zip(*data["lr"])

        # Plot losses
        #plt.plot(x_tr, ce_tr, 'lightblue')
        yhat = savgol_filter(ce_tr, 51, 3) # window size 51, polynomial order 3
        plt.plot(x_tr, yhat, markers[i], linewidth=1.0, markersize=5)
        plt.plot(x_va, ce_va, markers[i], linewidth=1.0, markersize=5)


    # mIoU Loss
    plt.subplot(3, 1, 2)
    plt.ylabel('mIoU')
    plt.xlabel('Iterations')
    #plt.legend(['train','filtered','valid'])
    for i, data in enumerate(grid_data):
        x_tr, ce_tr, dice_tr = zip(*data['train'])
        x_va, ce_va, dice_va = zip(*data['valid'])
        # invert dice loss to mIoU
        dice_tr = [1. - d_loss for d_loss in dice_tr]
        dice_va = [1. - d_loss for d_loss in dice_va]

        #plt.plot(x_tr, dice_tr, 'lightblue')
        yhat = savgol_filter(dice_tr, 51, 3) # window size 51, polynomial order 3
        plt.plot(x_tr, yhat, markers[i], linewidth=1.0, markersize=5)
        plt.plot(x_va, dice_va, markers[i], linewidth=1.0, markersize=5)


    # Plot learning rate
    #plt.subplot(3, 1, 3)
    #plt.plot(x_lr, lr_tr, 'red')
    #plt.grid(color='gray', linestyle='-', linewidth=0.5)
    #plt.ylabel('Learning Rate')


    # Save plots
    if save_path:
        plt.savefig(os.path.join(save_path, str(fname + '.pdf')), dpi=200)



# =======================================
# Miscellaneous
# =======================================

def JSD(p, q):
    M = 0.5*(p + q)
    return 0.5*np.sum(np.multiply(p, np.log(p/M))) + 0.5*np.sum(np.multiply(q, np.log(q/M)))

# Calculate class weight balancing
def eval_weights(probs):
    weights = 1 / (np.log(1.02 + (probs)))
    return weights/np.max(weights)


# Load test image/mask samples
# Load test image data subimages [NCWH]
def load_test_samples(img_path, mask_path, n_ch=1, n_classes=11, palette=None):
    
    if palette is None:
        palette = defaults.palette_rgb

    assert os.path.exists(img_path), print('Image path {} does not exist.'.format(img_path))
    if mask_path:
        assert os.path.exists(mask_path), print('Image path {} does not exist.'.format(mask_path))

    img_data = torch.as_tensor(get_image(img_path, n_ch), dtype=torch.float32)
    print('Test Image {} / Shape: {}'.format(img_path, img_data.shape))
    img_data = img_data.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    img_data = torch.reshape(img_data, (img_data.shape[0]*img_data.shape[1], n_ch, patch_size, patch_size) )
    print('Test Image Patches {} / Shape: {}'.format(img_data.shape[0], img_data.shape))

    # Load mask data subimages [NCWH]
    mask_data = None
    if mask_path:
        mask_data = torch.as_tensor(get_image(mask_path, 3), dtype=torch.int64)
        print('Test Mask {} / Shape: {}'.format(mask_path, mask_data.shape))
        mask_data = mask_data.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        mask_data = torch.reshape(mask_data, (mask_data.shape[0]*mask_data.shape[1], 3, patch_size, patch_size) )
        print('Test Mask Patches {} / Shape: {}'.format(mask_data.shape[0], mask_data.shape))

    return img_data, mask_data


# Compare experimental test outputs
def plot_sample_comparison(imgs, masks_true, masks_pred, exps, n_rows=40, offset=0):
    k = 0
    n_cols = 2 + len(masks_pred)
    fig, axes = plt.subplots(n_rows, n_cols, sharex='col', sharey='row', figsize=(n_cols*10, n_rows*10), dpi=100)
    plt.rcParams.update({'font.size': 10})

    # Plot sample subimages
    for i in range(n_rows):
        for j in range(n_cols):
            # show original subimage
            if j == 0:
                if imgs.ndim == 3:
                    axes[i,j].imshow(imgs[k + offset], cmap='gray')
                else:
                    axes[i,j].imshow(imgs[k + offset])
                axes[i,j].set_title('Image')
            # show true mask
            elif j == 1:
                axes[i,j].imshow(masks_true[k + offset], alpha=1.0)
                axes[i,j].set_title('Ground Truth')
            else:
                mask_pred = masks_pred[j - 2]
                axes[i,j].imshow(mask_pred[k + offset], alpha=1.0)
                axes[i,j].set_title('Experiment {}'.format(exps[j - 2]))
        k += 1

    plt.show()



# ===========================================================
# Mask Reconstruction
# ===========================================================

# Combines prediction mask tiles into full-sized mask
def reconstruct(tiles, full_width, full_height, n_classes, stride, offset=0):

    # Calculate reconstruction dimensions
    patch_size = tiles.shape[2]
    n_tiles = tiles.shape[0]
    n_strides_in_row = ( full_width ) // stride - 1
    n_strides_in_col = ( full_height ) // stride - 1

    # Calculate overlap
    olap_size = patch_size - stride

    # initialize full image numpy array
    full_mask = np.zeros((n_classes, full_height + offset, full_width), dtype=np.float32)

    # Create empty rows
    r_olap_prev = None
    r_olap_merged = None

    # row index (set to offset height)
    row_idx = offset

    for i in range(n_strides_in_col):
        # Get initial tile in row
        t_current = tiles[i*n_strides_in_row]
        r_current = np.zeros((n_classes, patch_size, full_width), dtype=np.float32)
        col_idx = 0
        # Step 1: Collate column tiles in row
        for j in range(n_strides_in_row):
            t_current_width = t_current.shape[2]
            if j < n_strides_in_row - 1:
                # Get adjacent tile
                t_next = tiles[i*n_strides_in_row + j + 1]
                # Extract right overlap of current tile
                olap_current = t_current[:, :, t_current_width - olap_size:t_current_width]
                # Extract left overlap of next (adjacent) tile
                olap_next = t_next[:, :, 0:olap_size]
                # Average the overlapping segment logits
                olap_current = torch.nn.functional.softmax(torch.tensor(olap_current), dim=0)
                olap_next = torch.nn.functional.softmax(torch.tensor(olap_next), dim=0)
                olap_merged = ( olap_current + olap_next ) / 2
                # Insert averaged overlap into current tile
                t_current[:, :, t_current_width - olap_size:t_current_width] = olap_merged
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
            r_olap_merged = ( r_olap_top + r_olap_prev ) / 2

        # Top row: crop by bottom overlap (to be averaged)
        if i == 0:
            # Crop current row by bottom overlap size
            r_current = r_current[:, 0:r_current_height - olap_size, :]
        # Otherwise: Merge top overlap with previous
        else:
            # Replace top overlap with averaged overlap in current row
            np.copyto(r_current[:, 0:olap_size, :], r_olap_merged)

        # Crop middle rows by bottom overlap
        if i > 0 and i < n_strides_in_col - 1:
            r_current = r_current[:, 0:r_current_height - olap_size, :]

        # Copy current row to full mask
        np.copyto(full_mask[:, row_idx:row_idx + r_current.shape[1], :], r_current)
        row_idx += r_current.shape[1]
        r_olap_prev = r_olap_bottom

    return full_mask