import numpy as np
import tifffile
from skimage.draw import ellipse
from pathlib import Path
from skimage.util import random_noise

if __name__ == "__main__":
    Path('data/a/').mkdir(parents=True, exist_ok=True)
    Path('data/a_GT/').mkdir(parents=True, exist_ok=True)
    Path('data/b/').mkdir(parents=True, exist_ok=True)
    Path('data/b_GT/').mkdir(parents=True, exist_ok=True)
    Path('data/c/').mkdir(parents=True, exist_ok=True)
    Path('data/c_GT/').mkdir(parents=True, exist_ok=True)

    inpute_img = np.zeros((1024, 1024))
    ellipse_list = [ellipse(160, 175, 80, 100),
                    ellipse(281, 175, 40, 100),
                    ellipse(250, 450, 100, 40)]

    # Create input Image
    inpute_img[ellipse_list[0]] = 1
    inpute_img[ellipse_list[1]] = 1
    inpute_img[ellipse_list[2]] = 1

    img = random_noise(inpute_img) * 255

    inpute_img[ellipse_list[0]] = 1
    inpute_img[ellipse_list[1]] = 2
    inpute_img[ellipse_list[2]] = 3

    for n in range(800):
        tifffile.imwrite(f"data/a/img_000{n}.tif", img.astype(np.uint8))
        tifffile.imwrite(f"data/b/img_000{n}.tif", img.astype(np.uint8))
        tifffile.imwrite(f"data/c/img_000{n}.tif", img.astype(np.uint8))
        tifffile.imwrite(f"data/a_GT/m_000{n}.tif", inpute_img.astype(np.uint8))
        tifffile.imwrite(f"data/b_GT/m_000{n}.tif", inpute_img.astype(np.uint8))
        tifffile.imwrite(f"data/c_GT/m_000{n}.tif", inpute_img.astype(np.uint8))
