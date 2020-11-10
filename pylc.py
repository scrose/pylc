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
File: pylc.py
"""
from utils.argparser import get_parser


def main():
    """
    Main application handler
    """
    # Get parsed input arguments
    parser = get_parser()
    args, unparsed = parser.parse_known_args()

    print(args, unparsed)

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print("\n\'{}\' is not a valid option.\n".format(unparsed[0]))
        parser.print_usage()
        exit(1)

    # execute processing function
    args.func(args)


if __name__ == "__main__":
    main()
