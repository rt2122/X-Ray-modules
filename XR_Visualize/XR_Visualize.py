from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import Tuple, List
from itertools import cycle


def get_ax(rows: int = 1, cols: int = 1, size: int = 12, shape: Tuple[int] = None
           ) -> matplotlib.axes.Axes:
    """
    Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    """
    if shape is None:
        shape = (size * cols, size * rows)
    else:
        shape = (shape[0] * size, shape[1] * size)
    _, ax = plt.subplots(rows, cols, figsize=shape)
    return ax


def get_bbox_picture(x0: float, y0: float, x1: float, y1: float, shape: Tuple[int]) -> np.ndarray:
    """
    Return numpy array with rectangle.
    """
    from skimage.draw import rectangle_perimeter
    bbox = np.zeros(shape)
    coords = rectangle_perimeter((y0, x0), (y1, x1), shape=shape)
    bbox[coords] = 1
    return bbox


def show_history(ax: matplotlib.axes.Axes, df: pd.DataFrame, metrics: List = None,
                 epochs: list = None, find_min: str = None, find_max: str = None) -> None:
    if ax is None:
        ax = get_ax()
    colors = 'bgrcmyk'

    if epochs is not None:
        df = df[epochs]
    epochs = df['epoch']

    if metrics is None:
        metrics = [k for k in list(df) if k != 'epoch']
    n_metr = len(metrics)
    metrics = list(filter(lambda x: not x.startswith('val'), metrics))
    val_flag = n_metr != len(metrics)

    for metric, c in zip(metrics, cycle(colors)):
        s, = ax.plot(epochs, df[metric], c=c)
        s.set_label(metric)

        if val_flag:
            metric = 'val_' + metric
            s, = ax.plot(epochs, df[metric], c=c, linestyle=':')
            s.set_label(metric)

    if find_min is not None:
        m = df[find_min].argmin()
        v = ax.axvline(m, c='r')
        v.set_label(f"Minimum of {find_min} at {m}")
    if find_max is not None:
        m = df[find_max].argmax()
        v = ax.axvline(m, c='r')
        v.set_label(f"Maximum of {find_max} at {m}")

    ax.set_xticks(epochs[::5])
    ax.grid()
    ax.legend()
