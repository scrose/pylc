"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
An evaluation of deep learning semantic segmentation for
land cover classification of oblique ground-based photography
MSc. Thesis 2020.
<http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Model Test
File: test.py
"""

import torch
from tqdm import tqdm
import utils.tools as utils
from config import cf
from utils.extract import Extractor
from utils.evaluate import Evaluator
from models.model import Model


def tester(args):
    """
    Apply model to input image(s) to generate segmentation maps.

    Parameters
    ----------
    args: dict
        User-defined options.
    """

    # Load pretrained model for testing or evaluation
    # Model file path is defined in user settings.
    model = Model(args).load(args.model)
    # self.net = torch.nn.DataParallel(self.net)
    model.net.eval()

    # get test file(s) - returns list of filenames
    files = utils.collate(args.img, args.mask)

    # initialize extractor, evaluator
    extractor = Extractor(args)
    evaluator = Evaluator(args)

    for f_idx, fpair in enumerate(files):

        # get image and associated mask data (if provided)
        if type(fpair) == dict and 'img' in fpair and 'mask' in fpair:
            img_file = fpair.get('img')
            mask_file = fpair.get('mask')
        else:
            img_file = fpair
            mask_file = None

        # extract image tiles (image is resized and cropped to fit tile size)
        img_tiles = extractor.load(img_file, mask_file).extract(fit=True, stride=cf.tile_size//2).get_data()
        img_loader, n_batches = img_tiles.loader(batch_size=args.batch_size)

        # get extraction metadata
        meta = extractor.meta.get_extract_meta()

        # apply model to input tiles
        with torch.no_grad():
            # get model outputs
            results = []
            for i, (tile, _) in tqdm(enumerate(img_loader), total=n_batches, desc="Segmentation: ", unit=' batches'):
                results.append(model.test(tile))
                model.iter += 1

        # load results into evaluator
        if mask_file:
            evaluator.load(results, meta, utils.get_image(mask_file, 3))
            # Evaluate prediction against ground-truth
            # (skip if only global/aggregated metrics requested)
            if not args.global_metrics:
                print("\nStarting evaluation of outputs ... ")
                evaluator.metrics.evaluate()
                evaluator.save_metrics()
        else:
            evaluator.load(results, meta)

        # save full-sized predicted mask image to file
        model.evaluator.save_image()

        # save unnormalized models outputs (i.e. raw logits) to file (if requested)
        if args.save_raw_output:
            model.evaluator.save_logits()
            print("Model output data saved to \n\t{}.".format(model.evaluator.output_path))

        # Reset evaluator
        model.evaluator.reset()

    # Compute global metrics
    if args.global_metrics:
        evaluator.metrics.evaluate(aggregate=True)
        evaluator.save_metrics()

