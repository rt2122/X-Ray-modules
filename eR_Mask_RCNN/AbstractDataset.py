import torch
import numpy as np
import os
from typing import Tuple, Dict, List
from pytorch_References import utils
from abs import abstractmethod
from astropy.WCS import WCS
from XR_Coord import wcs_from_fits


class AbstractDataset(torch.utils.data.Dataset):
    """Abstract class representing dataset.

    To subclass it, one need to overwrite :meth:`load_img` and :meth:`load_target`.
    These methods should use defined attributes:
    :param path: path for all dataset data.
    :type path: str
    :param sets: list of names of directories, from which data would be loaded.
    :type sets: List[str]
    :param files_dict: dict of files within each directory representing its meaning.
    For example, {"z": "img_z.fits"}.
    :type files_dict: Dict[str, str]
    """

    path: str
    sets: List[str]
    files_dict: Dict[str, str]
    data_loader: torch.utils.data.DataLoader

    @abstractmethod
    def __init__(self, **kwargs):
        """__init__.

        :param kwargs:
        """
        self.__dict__.update(kwargs)

    @abstractmethod
    def load_img(self, idx: int) -> np.ndarray:
        """load image by index.

        :param idx: index of image to load.
        :type idx: int
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def load_target(self, idx: int) -> np.ndarray:
        """load target by index.

        :param idx: index of target to load.
        :type idx: int
        :rtype: np.ndarray
        """
        pass

    def prepare_data(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Create DataLoader for this dataset.

        :param batch_size:
        :type batch_size: int
        :rtype: torch.utils.data.DataLoader
        """
        self.data_loader = torch.utils.data.DataLoader(self, batch_size=batch_size,
                                                       shuffle=True, num_workers=4,
                                                       collate_fn=utils.collate_fn)
        return self.data_loader

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = self.load_img(idx)
        target = self.load_target(idx)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.sets)

    def get_wcs(self, idx: int) -> WCS:
        """Extract WCS from one of the files for this index.

        :param idx: index of image.
        :type idx: int
        :rtype: WCS
        """
        path = os.path.join(self.path, self.sets[idx], self.files_dict.values()[0])
        return wcs_from_fits(path)
