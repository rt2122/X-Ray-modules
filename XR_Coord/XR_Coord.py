from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple


def wcs_from_fits(filename: str, index: int = 0, extract_data: bool = False) -> WCS:
    """
    Extract WCS object (and data as arrays) from fits file.
    """
    w = None
    d = []
    with fits.open(filename) as hdul:
        w = WCS(hdul[index].header)
        if extract_data:
            for i in range(len(hdul)):
                d.append(np.array(hdul[i].data))
    if extract_data:
        return w, d
    return w


def figure_wcs(w: WCS, figsize: Tuple[int] = (10, 10)):
    """
    Create figure and axis with WCS projection.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=w)
    ra = ax.coords[0]
    ra.set_format_unit('degree')
    return fig, ax


def show_fits(filename: str, fun: callable = None, data_index: int = 0,
              figsize: Tuple[int] = (10, 10), vmin=None, vmax=None, sparse: bool = False):
    """
    Show data from fits file with its WCS projection.
    """
    w, d = wcs_from_fits(filename, extract_data=True)
    if fun is not None:
        d = fun(d)
    fig, ax = figure_wcs(w, figsize=figsize)

    if not sparse:
        ax.imshow(d[data_index], cmap=plt.get_cmap('viridis'),
                  interpolation='nearest', vmin=vmin, vmax=vmax)
    else:
        ax.spy(d[data_index], cmap=plt.get_cmap('viridis'))
    return fig, ax
