"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
An evaluation of deep learning semantic segmentation for
land cover classification of oblique ground-based photography
MSc. Thesis 2020.
<http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Application
File: pylc.py
"""
import os
from utils.argparse import get_parser
from config import defaults


def main():
    """
    Main application handler
    """
    # Get parsed input arguments
    parser = get_parser()
    args, unparsed = parser.parse_known_args()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print("\n\'{}\' is not a valid option.\n".format(unparsed[0]))
        parser.print_usage()
        exit(1)

    # ensure data directories exist in project root
    dirs = [defaults.root, defaults.db_dir, defaults.save_dir, defaults.model_dir, defaults.output_dir]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    # execute processing function
    args.func(args)


if __name__ == "__main__":
    main()
