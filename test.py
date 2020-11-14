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
from config import defaults
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
    img_path = args.img
    mask_path = args.mask
    model_path = args.model
    scale = args.scale
    aggregate_metrics = args.aggregate_metrics
    save_logits = args.save_logits

    # Load model for testing/evaluation
    model = Model(args).load(model_path)
    model.print_settings()
    # self.net = torch.nn.DataParallel(self.net)
    model.net.eval()

    # get test file(s) - returns list of filenames
    files = utils.collate(img_path, mask_path)

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
        img_tiles = extractor.load(img_file).extract(
            fit=True,
            stride=defaults.tile_size // 2,
            scale=scale
        ).get_data()

        # get data loader
        img_loader, n_batches = img_tiles.loader(batch_size=1)

        if not input("\nContinue with segmentation? (Enter \'Y\' or \'y\' for Yes): ") in ['Y', 'y']:
            print('Stopped.')
            exit(0)

        # apply model to input tiles
        with torch.no_grad():
            # get model outputs
            results = []
            for i, (tile, _) in tqdm(enumerate(img_loader), total=n_batches, desc="Segmentation: ", unit=' batches'):
                result = model.test(tile)
                results.extend(result)
                model.iter += 1

        # load results into evaluator
        results = utils.reconstruct(results, extractor.get_meta())
        # - save full-sized predicted mask image to file
        if mask_file:
            evaluator.load(
                results,
                extractor.get_meta(),
                mask_true_path=mask_file,
                scale=scale
            ).save_image()
            # Evaluate prediction against ground-truth
            # - skip if only global/aggregated requested
            if not aggregate_metrics:
                print("\nStarting evaluation ... ")
                evaluator.evaluate().save_metrics()
        else:
            evaluator.load(results, extractor.get_meta()).save_image()

        # save unnormalized models outputs (i.e. raw logits) to file (if requested)
        if save_logits:
            evaluator.save_logits()
            print("Model output data saved to \n\t{}.".format(model.evaluator.output_path))

        # Reset evaluator
        evaluator.reset()

    # Compute global metrics
    if aggregate_metrics:
        evaluator.evaluate(aggregate=True)
        evaluator.save_metrics()
