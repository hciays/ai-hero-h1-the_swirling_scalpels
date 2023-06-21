import time
from torch.utils.data import Dataset
import tifffile
from pathlib import Path
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.draw import ellipse
from scipy.ndimage import distance_transform_edt, binary_closing, grey_closing, generate_binary_structure, \
    binary_erosion
from skimage.morphology import disk
from skimage import measure


class CellDataset(Dataset):
    def __init__(self, root_dir, border_core=True, split="train", transform=None, local_test=False, cache=True):

        self.transform = transform
        self.border_core = border_core

        self.local_test = local_test

        root_dir = Path(root_dir)
        if split == "train":
            self.img_files = sorted(list(root_dir.glob(r'[ab]' + "/*.tif")))
            self.mask_files = sorted(list(root_dir.glob(r'[ab]' + "_GT/*.tif")))
        elif split == "val":
            self.img_files = sorted(list(root_dir.glob(r'[c]' + "/*.tif")))
            self.mask_files = sorted(list(root_dir.glob(r'[c]' + "_GT/*.tif")))
        elif split == "test":
            self.img_files = sorted(list(root_dir.glob(r'[de]' + "/*.tif")))
            self.mask_files = None

        self.cache = ([None] * len(self.img_files)) if cache else None
        self.new_pp = True

    def __getitem__(self, idx):
        img = tifffile.imread(self.img_files[idx])
        mask = tifffile.imread(self.mask_files[idx]).astype(np.float32) if self.mask_files else None

        orig_size = img.shape
        file_name = '/'.join(str(self.img_files[idx]).split('/')[-2:])

        if self.border_core:
            if self.mask_files:
                # convert masks to border core representation where each instance gets label 1 for the core part of the
                # instance and label 2 for the border part of the instance

                # Create an array to store eroded instances
                eroded_instances = np.zeros_like(mask)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Define the erosion kernel
                for instance_label in np.unique(mask)[1:]:  # Exclude background label 0
                    instance_mask = (mask == instance_label).astype(np.uint8)
                    eroded_instance = cv2.erode(instance_mask, kernel, iterations=4)
                    eroded_instances += eroded_instance

                mask = np.where(eroded_instances == 1, 1, 2 * mask.clip(0, 1))  # 1 = core, 2 = border

        if self.transform is not None:
            if self.mask_files:
                transformed = self.transform(image=img, mask=mask)
            else:
                transformed = self.transform(image=img)
            img = transformed['image']
            if self.mask_files:
                mask = transformed['mask'].long()

        if self.mask_files and not self.border_core:
            mask = mask.unsqueeze(0)

        if self.local_test:
            return img, mask if mask is not None else 1, orig_size, file_name
        else:
            return img, mask if mask else 1, orig_size, file_name

    def __len__(self):
        return len(self.img_files)

    def _new_pp(self,mask):
        mask = np.zeros_like(mask)
        ellipse_list = [ellipse(160, 175, 80, 100),
                        ellipse(281, 175, 40, 100),
                        ellipse(250, 450, 100, 40)]

        # Create input Image
        mask[ellipse_list[0]] = 1
        mask[ellipse_list[1]] = 2
        mask[ellipse_list[2]] = 3

        result_image_neighbors = np.zeros_like(mask)
        result_image_cell = np.zeros_like(mask)

        for instance_label in np.unique(mask)[1:]:  # Exclude background label 0
            microbe_coords = np.where(mask == instance_label)
            # cell
            cell = np.zeros_like(mask)
            cell[microbe_coords] = 1
            result_image_cell[microbe_coords] = distance_transform_edt(cell)[microbe_coords]

            # b)
            tmp_result_image_neighbors = mask
            tmp_result_image_neighbors[microbe_coords] = 0
            tmp_result_image_neighbors = tmp_result_image_neighbors != 0
            image = 1 - tmp_result_image_neighbors

            # c)
            dist_img = distance_transform_edt(image)

            # d)
            d = np.zeros_like(mask)
            d[microbe_coords] = dist_img[microbe_coords]

            # e)
            e = np.zeros_like(mask)
            min = np.min(d)
            max = np.max(d)
            e[microbe_coords] = 1 - ((d[microbe_coords] - min) * (1.0 / (max - min)))

            # f)
            result_image_neighbors[microbe_coords] = e[microbe_coords]

        close_img = np.zeros_like(result_image_neighbors)

        # g)
        for ellipse_elm in np.unique(mask)[1:]:
            nucleus = np.zeros_like(mask)
            nucleus[ellipse_elm] = mask[ellipse_elm]
            nucleus = binary_closing(nucleus, disk(3))
            close_img[nucleus] = True

        label_bottom_hat = binary_closing(close_img, disk(3)).astype(bool) ^ close_img.astype(bool)
        label_closed = (~close_img.astype(bool)) & label_bottom_hat

        label_closed = measure.label(label_closed.astype(np.uint8))
        props = measure.regionprops(label_closed)
        label_closed_corr = (label_closed > 0).astype(np.float32)
        for i in range(len(props)):
            if props[i].minor_axis_length >= 3:
                single_gap = label_closed == props[i].label
                single_gap_border = single_gap ^ binary_erosion(single_gap, generate_binary_structure(2, 1))
                label_closed_corr[single_gap] = 1
                label_closed_corr[single_gap_border] = 0.8  # gets scaled later to 0.84

        result_image_neighbors = np.maximum(result_image_neighbors, label_closed_corr.astype(result_image_neighbors.dtype))
        result_image_neighbors = np.maximum(result_image_neighbors, label_closed.astype(result_image_neighbors.dtype))

        # h)
        result_image_neighbors = 1 / np.sqrt(0.65 + 0.5 * np.exp(-11 * (result_image_neighbors - 0.75))) - 0.19
        result_image_neighbors = np.clip(result_image_neighbors, 0, 1)
        result_image_neighbors = grey_closing(result_image_neighbors, size=(3, 3))

        #fig, ax = plt.subplots(1,2)
        #ax[0].imshow(result_image_neighbors, cmap=plt.cm.gray)
        #ax[1].imshow(result_image_cell, cmap=plt.cm.gray)
        #plt.savefig('Test.png')

        return np.ascontiguousarray(result_image_cell), np.ascontiguousarray(result_image_neighbors)


def train_transform():
    transform = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(0, 1),
        ToTensorV2()
    ], additional_targets={'mask1': 'mask'})

    return transform


def val_transform(imgsz=256):
    transform = A.Compose([
        A.Resize(imgsz, imgsz, interpolation=cv2.INTER_NEAREST),
        A.Normalize(0, 1),
        ToTensorV2()
    ], additional_targets={'mask1': 'mask'})

    return transform
