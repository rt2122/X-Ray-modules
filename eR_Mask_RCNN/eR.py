import torch
import os
import numpy as np
from eR_Mask_RCNN.transforms import normalize_2d, get_bbox_from_mask
from eR_Mask_RCNN import cat2masks
from astropy.io.fits import getdata


def load_image(path: str, raw_size: int, grid_size: int):
    '''
    Load image from set path. Raw size - initial size of x,y dimensions.
    Grid size - width of maps cut after overlaying.
    '''
    exp = getdata(os.path.join(path, 'exp_tile.fits.gz'), memmap=False)
    tile = getdata(os.path.join(path, 'tile.fits.gz'), memmap=False)

    image = np.zeros([raw_size, raw_size, 1], dtype=np.float32)
    image[:, :, 0] = tile / exp
    image = np.nan_to_num(np.log(image))
    idx = slice(grid_size, raw_size - grid_size)  # remove grid
    image = image[idx, idx]
    image[:, :, 0] = normalize_2d(image[:, :, 0])
    return image


class eR_Dataset(torch.utils.data.Dataset):

    def __init__(self, path, transforms, raw_size: int = 3240, grid_size: int = 270):
        self.raw_size = raw_size
        self.grid_size = grid_size
        self.size = raw_size - 2 * grid_size

        self.transforms = transforms
        sets = os.listdir(path)
        self.sets = sorted(sets, key=lambda x: int(x[4:]))
        self.path = path

    def __getitem__(self, idx):
        setpath = os.path.join(self.path, self.sets[idx])
        image = load_image(os.path.join(self.path, setpath), self.raw_size, self.grid_size)

        # Load gt
        catpath = os.path.join(self.path, setpath, 'objects.csv')

        boxes = []
        masks, labels = cat2masks(catpath, raw_size=self.raw_size, grid_size=self.grid_size)
        sources = masks.shape[-1]
        for i in range(masks.shape[-1]):
            boxes.append(get_bbox_from_mask(masks[:, :, i]))

        masks = torch.as_tensor(masks, dtype=torch.uint8).permute(2, 0, 1)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int32)
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
