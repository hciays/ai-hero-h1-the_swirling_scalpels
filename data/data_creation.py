import numpy as np
import tifffile
import cv2

if __name__ == "__main__":

    image_shape = (1024, 1024)
    train = np.zeros(image_shape, dtype=np.uint8)
    val = np.zeros(image_shape, dtype=np.uint8)
    test = np.zeros(image_shape, dtype=np.uint8)

    train_m = np.array([[100, 100], [300, 800], [800, 400], [600, 100]], dtype=np.int32)
    val_m = np.array([[200, 200], [500, 200], [500, 500]], dtype=np.int32)
    test_m = np.array([[50, 500], [400, 300], [600, 700], [300, 900]], dtype=np.int32)

    cv2.fillPoly(train, [train_m], 255)
    cv2.fillPoly(val, [val_m], 255)
    cv2.fillPoly(test, [test_m], 255)

    tifffile.imwrite("a/img_0001.tif", train, compress=6)
    tifffile.imwrite("b/img_0001.tif", val, compress=6)
    tifffile.imwrite("c/img_0001.tif", test, compress=6)
    tifffile.imwrite("a_GT/m_0001.tif", train, compress=6)
    tifffile.imwrite("b_GT/m_0001.tif", val, compress=6)
    tifffile.imwrite("c_GT/m_0001.tif", test, compress=6)