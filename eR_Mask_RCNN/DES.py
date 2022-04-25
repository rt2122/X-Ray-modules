import torch

import os
import numpy as np
from astropy.io.fits import getdata
from astropy.io import fits
from eR_Mask_RCNN.transforms import get_bbox_from_mask, normalize_2d, Compose


class DES_Dataset(torch.utils.data.Dataset):
    SIZE = 512

    def __init__(self, path: str, transforms: Compose, add_norm: bool = True,
                 clone_channels: str = None):
        self.path = path
        self.transforms = transforms
        sets = os.listdir(path)

        sets = sorted(sets, key=lambda x: int(x[4:]))
        self.sets = sets
        self.add_norm = add_norm
        self.clone_channels = clone_channels

    def __getitem__(self, idx):
        setpath = os.path.join(self.path, self.sets[idx])

        g = getdata(os.path.join(setpath, "img_g.fits"), memmap=False)
        r = getdata(os.path.join(setpath, "img_r.fits"), memmap=False)
        z = getdata(os.path.join(setpath, "img_z.fits"), memmap=False)

        image = np.zeros([DES_Dataset.SIZE, DES_Dataset.SIZE, 3], dtype=np.float32)

        # zscore normalize
        I = (z + r + g) / 3.0 # noqa E741
        Isigma = I * np.mean([np.std(g), np.std(r), np.std(z)])
        z = (z - np.mean(z)) / Isigma
        r = (r - np.mean(r)) / Isigma
        g = (g - np.mean(g)) / Isigma

        max_RGB = np.percentile([z, r, g], 99.995)
        # avoid saturation
        r = r / max_RGB
        g = g / max_RGB
        z = z / max_RGB

        if self.add_norm:
            z = normalize_2d(z)
            r = normalize_2d(r)
            g = normalize_2d(g)

        image[:, :, 0] = z  # red
        image[:, :, 1] = r  # green
        image[:, :, 2] = g  # blue

        if self.clone_channels is not None:
            clone = None
            if self.clone_channels == 'z':
                clone = z
            elif self.clone_channels == 'r':
                clone = r
            elif self.clone_channels == 'g':
                clone = g
            if clone is not None:
                image = np.dstack([clone] * 3)

        with fits.open(os.path.join(setpath, 'masks.fits'),
                       memmap=False, lazy_load_hdus=False) as hdul:
            sources = len(hdul)
            data = [hdu.data / np.max(hdu.data) for hdu in hdul]
            labels = [hdu.header["CLASS_ID"] for hdu in hdul]

        thresh = [0.005 if i == 1 else 0.08 for i in labels]
        masks = np.zeros([sources, DES_Dataset.SIZE, DES_Dataset.SIZE], dtype=np.uint8)
        boxes = []
        for i in range(sources):
            masks[i, :, :][data[i] > thresh[i]] = 1
            boxes.append(get_bbox_from_mask(masks[i]))

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((sources,), dtype=torch.int64)

        masks = masks[area != 0, :]
        boxes = boxes[area != 0, :]
        labels = labels[area != 0]
        iscrowd = iscrowd[area != 0]
        area = area[area != 0]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.sets)
