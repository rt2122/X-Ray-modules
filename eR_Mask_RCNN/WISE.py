import numpy as np
import torch
import os
from eR_Mask_RCNN.transforms import Compose, discretize_with_quantile, normalize_2d
from astropy.io.fits import getdata


def wise_normalize(w: np.ndarray, w_inv: np.ndarray) -> np.ndarray:
    w = np.sqrt(w_inv) * w
    w = np.clip(w, a_min=-10, a_max=10)
    return w


class WISE_Dataset(torch.utils.data.Dataset):
    SIZE = 2048

    def __init__(self, path: str, transforms: Compose, add_norm: bool = True,
                 clone_channels: str = '122', add_flat: bool = False, res_mode: bool = False):
        self.path = path
        self.transforms = transforms
        sets = os.listdir(path)

        sets = sorted(sets)
        self.sets = sets
        self.add_norm = add_norm
        if len(clone_channels) != 3:
            clone_channels = clone_channels[0] * 3
        self.clone_channels = clone_channels
        self.add_flat = add_flat
        self.res_mode = res_mode

    def __getitem__(self, idx: int):
        cset = self.sets[idx]
        setpath = os.path.join(self.path, cset)
        image = np.zeros([self.SIZE, self.SIZE, 3], dtype=np.float32)

        if self.res_mode:
            w1 = getdata(os.path.join(setpath, "w1_res.fits.gz"), memmap=False)
            w2 = getdata(os.path.join(setpath, "w2_res.fits.gz"), memmap=False)
        else:
            w1 = getdata(os.path.join(setpath, f"unwise-{cset}-w1-img-m.fits"), memmap=False)
            w2 = getdata(os.path.join(setpath, f"unwise-{cset}-w2-img-m.fits"), memmap=False)

        w1_inv = getdata(os.path.join(setpath, f"unwise-{cset}-w1-invvar-m.fits.gz"), memmap=False)
        w2_inv = getdata(os.path.join(setpath, f"unwise-{cset}-w2-invvar-m.fits.gz"), memmap=False)

        w1 = wise_normalize(w1, w1_inv)
        w2 = wise_normalize(w2, w2_inv)

        if self.add_flat:
            w1 = discretize_with_quantile(w1)
            w2 = discretize_with_quantile(w2)

        if self.add_norm:
            w1 = normalize_2d(w1)
            w2 = normalize_2d(w2)

        img_dict = {'1': w1, '2': w2}
        for i in range(3):
            image[:, :, i] = img_dict[self.clone_channels[i]]

        if self.transforms is not None:
            image, _ = self.transforms(image, None)

        gt = {"image_id": torch.tensor([idx], dtype=torch.int32), "set": self.sets[idx],
              "boxes": torch.tensor([]), "masks": torch.tensor([])}

        return image, gt

    def __len__(self):
        return len(self.sets)

    def wcs_path(self, idx: int) -> str:
        """
        Get path of file for extracting WCS.
        """
        return os.path.join(self.path, self.sets[idx], "w1_res.fits.gz")
