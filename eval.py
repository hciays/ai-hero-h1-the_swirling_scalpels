from acvl_utils.instance_segmentation.instance_matching import compute_all_matches
import numpy as np
from argparse import ArgumentParser
import tifffile
import glob
import os


def load_fn(file_path):
    return tifffile.imread(file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="./pred",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/hkfs/work/workspace/scratch/hgf_pdv3669-health_train_data/resized_256",
    )
    args = parser.parse_args()

    pred_list = sorted(glob.glob(os.path.join(args.pred_dir, '*/*.tif')))
    gt_list = sorted(glob.glob(os.path.join(args.gt_dir, 'c_GT_resized_256/*.tif')))

    # mean instance dice for each image over all images
    print('Computing Scores...')
    results = compute_all_matches(gt_list, pred_list, load_fn, num_processes=12)
    # print('Done')
    score = np.array([np.array(([r[2] for r in results[i]])).mean() for i in range(len(results))]).mean()
    print('Mean Instance Dice:', score)
