"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
An evaluation of deep learning semantic segmentation for
land cover classification of oblique ground-based photography
MSc. Thesis 2020.
<http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Application
File: app.py
"""

from preprocess import preprocess
from train import train
from test import test
from config import cf


def main():
    """
    Main application handler
    """

    # --- Preprocessing ---
    if cf.mode in (cf.EXTRACT, cf.AUGMENT, cf.MERGE, cf.GRAYSCALE):
        preprocess()

    # --- Model training ---
    elif cf.mode == cf.TRAIN:
        train()

    # --- Model training ---
    elif cf.mode == cf.TEST:
        test()

    else:
        print("\'{}\' is not a valid application mode.".format(cf.mode))
        cf.parser.print_usage()


if __name__ == "__main__":

    main()
