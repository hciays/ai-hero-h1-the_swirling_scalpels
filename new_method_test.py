import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.segmentation import watershed
from skimage import measure
from skimage.feature import peak_local_max, canny
from skimage.morphology import binary_closing
import argparse

def distance_postprocessing(border_prediction, cell_prediction):
    """ Post-processing for distance label (cell + neighbor) prediction.

    :param border_prediction: Neighbor distance prediction.
        :type border_prediction:
    :param cell_prediction: Cell distance prediction.
        :type cell_prediction:
    :param args: Post-processing settings (th_cell, th_seed, n_splitting, fuse_z_seeds).
        :type args:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
    :return: Instance segmentation mask.
    """

    args = {

    }

    # Smooth predictions slightly + clip border prediction (to avoid negative values being positive after squaring)
    sigma_cell = 1.0
    sigma_border = 0.5

    cell_prediction = gaussian_filter(cell_prediction, sigma=sigma_cell)
    border_prediction = gaussian_filter(border_prediction, sigma=sigma_border)
    border_prediction = np.clip(border_prediction, 0, 1)

    th_seed = args.th_seed
    th_cell = args.th_cell
    th_local = 0.25

    # Get mask for watershed
    mask = cell_prediction > th_cell

    # Get seeds for marker-based watershed
    borders = np.tan(border_prediction ** 2)
    borders[borders < 0.05] = 0
    borders = np.clip(borders, 0, 1)
    cell_prediction_cleaned = (cell_prediction - borders)
    seeds = cell_prediction_cleaned > th_seed
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    areas = []
    for i in range(len(props)):
        areas.append(props[i].area)
    if len(areas) > 0:
        min_area = 0.10 * np.mean(np.array(areas))
    else:
        min_area = 0
    min_area = np.maximum(min_area, 4)

    for i in range(len(props)):
        if props[i].area <= min_area:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Avoid empty predictions (there needs to be at least one cell)
    while np.max(seeds) == 0 and th_seed > 0.05:
        th_seed -= 0.1
        seeds = cell_prediction_cleaned > th_seed
        seeds = measure.label(seeds, background=0)
        props = measure.regionprops(seeds)
        for i in range(len(props)):
            if props[i].area <= 4:
                seeds[seeds == props[i].label] = 0
        seeds = measure.label(seeds, background=0)

    if args.fuse_z_seeds:
        seeds = seeds > 0
        kernel = np.ones(shape=(3, 1, 1))
        seeds = binary_closing(seeds, kernel)
        seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)

    if args.apply_merging and np.max(prediction_instance) < 255:
        # Get borders between touching cells
        label_bin = prediction_instance > 0
        pred_boundaries = cv2.Canny(prediction_instance.astype(np.uint8), 1, 1) > 0
        pred_borders = cv2.Canny(label_bin.astype(np.uint8), 1, 1) > 0
        pred_borders = pred_boundaries ^ pred_borders
        pred_borders = measure.label(pred_borders)
        for border_id in get_nucleus_ids(pred_borders):
            pred_border = (pred_borders == border_id)
            if np.sum(border_prediction[pred_border]) / np.sum(
                    pred_border) < 0.075:  # very likely splitted due to shape
                # Get ids to merge
                pred_border_dilated = binary_dilation(pred_border, np.ones(shape=(3, 3), dtype=np.uint8))
                merge_ids = get_nucleus_ids(prediction_instance[pred_border_dilated])
                if len(merge_ids) == 2:
                    prediction_instance[prediction_instance == merge_ids[1]] = merge_ids[0]
        prediction_instance = measure.label(prediction_instance)

    return np.squeeze(prediction_instance.astype(np.uint16)), np.squeeze(borders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Hero Hackathon 2023 - Inference')
    parser.add_argument('--apply_clahe', '-acl', default=False, action='store_true', help='CLAHE pre-processing')
    parser.add_argument('--apply_merging', '-am', default=False, action='store_true', help='Merging post-processing')
    parser.add_argument('--artifact_correction', '-ac', default=False, action='store_true', help='Artifact correction')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--fuse_z_seeds', '-fzs', default=False, action='store_true',
                        help='Fuse seeds in axial direction')
    parser.add_argument('--multi_gpu', '-mgpu', default=True, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--n_splitting', '-ns', default=40, type=int,
                        help='Cell amount threshold to apply splitting post-processing (3D)')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save some raw predictions')
    parser.add_argument('--scale', '-sc', default=1, type=float, help='Scale factor')
    parser.add_argument('--subset', '-s', default='01+02', type=str, help='Subset to evaluate on')
    parser.add_argument('--th_cell', '-tc', default=0.07, type=float, help='Threshold for adjusting cell size')
    parser.add_argument('--th_seed', '-ts', default=0.45, type=float, help='Threshold for seeds')
    args = parser.parse_args()

    image_shape = (256, 256)
    train = np.zeros(image_shape, dtype=np.uint8)
    val = np.zeros(image_shape, dtype=np.uint8)

    train_m = np.array([[100, 100], [300, 800], [800, 400], [600, 100]], dtype=np.int32)
    val_m = np.array([[200, 200], [500, 200], [500, 500]], dtype=np.int32)

    cv2.fillPoly(train, [train_m], 1)
    cv2.fillPoly(val, [val_m], 1)

    start = time.time()
    distance_postprocessing(train, val)
    end = time.time()
    print("Time ", end - start)
