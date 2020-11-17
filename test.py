"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
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
from config import defaults, Parameters
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
    # trained model path
    model_path = args.model

    # load parameters
    params = Parameters(args)

    # Load model for testing/evaluation
    model = Model().load(model_path)
    model.print_settings()
    model.net.eval()

    # get test file(s) - returns list of filenames
    files = utils.collate(args.img, args.mask)

    # initialize extractor, evaluator
    extractor = Extractor(model.meta)
    evaluator = Evaluator(model.meta)

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
            scale=params.scale
        ).get_data()

        # get data loader
        img_loader, n_batches = img_tiles.loader(
            batch_size=8,
            drop_last=False
        )

        # if not input("\nContinue with segmentation? (Enter \'Y\' or \'y\' for Yes): ") in ['Y', 'y']:
        #     print('Stopped.')
        #     exit(0)

        # apply model to input tiles
        with torch.no_grad():
            # get model outputs
            model_outputs = []
            for i, (tile, _) in tqdm(enumerate(img_loader), total=n_batches, desc="Segmentation: ", unit=' batches'):
                logits = model.test(tile)
                model_outputs += logits
                model.iter += 1

        # load results into evaluator
        results = utils.reconstruct(model_outputs, extractor.get_meta())
        # - save full-sized predicted mask image to file
        if mask_file:
            evaluator.load(
                results,
                extractor.get_meta(),
                mask_true_path=mask_file,
                scale=params.scale
            ).save_image()

            # Evaluate prediction against ground-truth
            # - skip if only global/aggregated requested
            if not params.aggregate_metrics:
                print("\nStarting evaluation ... ")
                evaluator.evaluate().save_metrics()
        else:
            evaluator.load(results, extractor.get_meta()).save_image()

        # save unnormalized models outputs (i.e. raw logits) to file (if requested)
        if args.save_logits:
            evaluator.save_logits(model_outputs)

        # Reset evaluator
        evaluator.reset()

    # Compute global metrics
    if args.aggregate_metrics:
        evaluator.evaluate(aggregate=True)
        evaluator.save_metrics()
