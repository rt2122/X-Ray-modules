"""This module contains augmentation classes and normalizaton method."""
import numpy as np
import torch

from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import Tuple, Dict, Optional
import random


class Compose(object):
    """Connect all augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(nn.Module):
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
                ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
                ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F.get_image_size(image)  # instead of _get_image_size # noqa: E501
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
                ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height = F.get_image_size(image)  # instead of _get_image_size # noqa: E501
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
                target["masks"] = target["masks"].flip(1)
        return image, target


def get_bbox_from_mask(mask: np.ndarray) -> Tuple[float, ...]:
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]


class Rotation90:
    """Rotate by the angle 90 n times"""

    def __call__(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
                 ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        n = random.randint(0, 4)
        image = torch.rot90(image, n, [1, 2])
        if target is not None:
            target["masks"] = torch.rot90(target["masks"], n, [1, 2])
            boxes = []
            for i in range(target['masks'].shape[0]):
                boxes.append(get_bbox_from_mask(target['masks'].numpy()[i]))
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        return image, target


class GaussianBlur(T.GaussianBlur):
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._backward_hooks = None
        self._forward_hooks = None
        self._forward_pre_hooks = None
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
                ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.gaussian_blur(image, self.kernel_size, self.sigma)
        image = normalize_image(image)
        return image, target


def detect_empty_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """
    Return array with indexes of non-empty boxes.
    """
    return torch.where(torch.logical_and((boxes[:, 2] - boxes[:, 0]) > 0,
                       (boxes[:, 3] - boxes[:, 1]) > 0))


class Clip:
    """
    Get smaller picture.
    """
    def __init__(self, small_size: int):
        self.small_size = small_size
    def __call__(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
                 ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        big_size = image.shape[-1]
        # Chose random box
        n = np.random.randint(0, len(target["boxes"]))
        box = target["boxes"][n]

        # Random area should include this box
        shift = box[:2].int()
        for i in range(2):
            if shift[i] + self.small_size >= big_size:
                shift[i] = big_size - self.small_size - 1

        slices = [slice(shift[i], shift[i] + self.small_size) for i in range(2)]
        image = image[:, slices[0], slices[1]]
        if target is not None:
            masks = target["masks"]
            masks = [mask[slices[0], slices[1]] for mask in masks]
            non_zero_masks = torch.tensor([mask.any() for mask in masks])
            non_zero_masks = torch.where(non_zero_masks)
            masks = torch.stack(masks)[non_zero_masks]
            target["masks"] = masks

            boxes = []
            for i in range(target['masks'].shape[0]):
                boxes.append(get_bbox_from_mask(target['masks'].numpy()[i]))
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target["boxes"] = boxes
            target["labels"] = target["labels"][non_zero_masks]
            target["iscrowd"] = target["iscrowd"][non_zero_masks]
            target["area"] = target["area"][non_zero_masks]

            if len(boxes.shape) > 1:
                empty = detect_empty_boxes(boxes)
                if len(empty) > 0:
                    for k, v in target.items():
                        if k != "image_id":
                            target[k] = v[empty]

        return image, target


def get_augmentation(yes_aug: bool = True, small_size: int = None):
    """
    Get augmentation (if yes_aug) or just transform to tensor.
    """
    transformations = []
    transformations.append(ToTensor())
    if small_size is not None:
        transformations.append(Clip(small_size))
    if yes_aug:
        transformations.append(RandomHorizontalFlip(0.5))
        transformations.append(RandomVerticalFlip(0.5))
        transformations.append(Rotation90())
        transformations.append(GaussianBlur(kernel_size=1, sigma=(2, 6)))
    return Compose(transformations)


def normalize_2d(a):
    """Normalize 2d image."""
    a -= a.min()
    a /= a.max()
    return a


def normalize_image(image):
    """Normalize image regardless of order of dimensions."""
    if image.shape[0] < 4:
        for i in range(image.shape[0]):
            image[i] = normalize_2d(image[i])
    else:
        for i in range(image.shape[-1]):
            image[:, :, i] = normalize_2d(image[:, :, i])
    return image
