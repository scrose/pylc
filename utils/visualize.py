# MLP training data preprocessing

import os, sys, glob, math
import random
import numpy as np
import torch
import h5py
from visdom import Visdom

# Visualization Tools
# -------------------

# Loss/Accuracy line plotter
# Reference:
# https://github.com/noagarcia/visdom-tutorial

class LinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        assert self.viz.check_connection(), 'Visdom connection could not be formed quickly.'
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
            X=np.array([x,x]),
            Y=np.array([y,y]),
            env=self.env,
            opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(
            X=np.array([x]),
            Y=np.array([y]),
            env=self.env,
            win=self.plots[var_name],
            name=split_name,
            update = 'append')

class Visualizer(object):
    '''displays image masks'''
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.views = {}
        assert self.viz.check_connection(timeout_seconds=3), \
        'No connection could be formed quickly'

    def view(self, img_data, view_name, title_name='', caption=''):
        '''
        Display mulitiple images in Visdom window
        img_array: Numpy array of images with shape NCHW
        view_name: name of the view windown in Visdom
        '''
        img_array = []
        # convert to RGB if needed
        for i, img in enumerate(img_data):
            # Collapse one-hot encoding
            if img.shape[0] == 9:
                img = np.argmax(img, axis=0).reshape(1, img.shape[1], img.shape[2])
            # Colourize mask data
            if img.shape[0] == 1:
                img = colourize(img)

            img_array += [img]

        # display image to window
        self.viz.images(
        img_array,
        win=view_name,
        env=self.env,
        opts=dict(caption=caption, store_history=True, title=title_name),
    )


# Plot colour-coded categories
def plot_mask_colours(colours, title, autotext):

    cell_width = 260
    cell_height = 30
    swatch_width = 48
    margin = 12
    topmargin = 40

    n = len(colours)
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

    for i, colour in enumerate(colours):
        row = i % nrows
        col = i // nrows
        y = row * cell_height
        label = ''

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        if (i < len(autotext)):
            label = ' (' + autotext[i]  + ') '

        ax.text(text_pos_x, y, mask_categories[colour] + label, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=colour, linewidth=18)

    return fig


# Show samples image patches from the database
def show_sample(n_rows, n_cols, offset, config):

    # Load hdf5 file
    hdf5_file = h5py.File(config.db_path, mode='r')
    img_patches = hdf5_file.get('train_img').value.astype(int)
    mask_patches = hdf5_file.get('train_mask').value

    # permute images axis to NWHC
    img_patches = np.moveaxis(img_patches, 1, -1)

    # Plot settings
    fig, axes = plt.subplots(n_rows, n_cols, sharex='col', sharey='row', figsize=(100, 400),
                             subplot_kw={'xticks': [], 'yticks': []})
    plt.rcParams.update({'font.size': 44})

    # Plot sample subimages
    for i in range(0, n_rows, 2):
        for j in range(n_cols):
            # Get sample patch & mask
            img_patch_sample = img_patches[offset + i + j]
            mask_patch_sample = mask_patches[offset + i + j]*30
            # collapse one-hot encoding to single channel
            mask_patch_sample = np.sum(mask_patch_sample, axis=0).astype(np.uint8)

            # add original image and mask to subplot array
            axes[i,j].imshow(img_patch_sample)
            axes[i,j].set_title('original')


            # pad mask to show crop and center region
            mask_patch_sample = np.pad(mask_patch_sample, pad_size)
            # add original image and mask to subplot array
            axes[i + 1,j].imshow(mask_patch_sample)

            axes[i + 1,j].set_title('mask')

    plt.show()
    hdf5_file.close()


# Read image and reverse channel order
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Read image and reverse channel order
def colourize(img, palette):
    img = img.squeeze()
    w = img.shape[0]
    h = img.shape[1]
    # make 3-channel image
    img = np.moveaxis(np.stack((img,)*3, axis=0), 0, -1).reshape(w*h, 3)
    # map categories to palette colours
    for i, value in enumerate(img):
        img[i] = palette[value[0]]
    img = np.moveaxis(img.reshape(w, h, 3), -1, 0)
    return img
