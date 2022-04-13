from matplotlib import pyplot as plt
import numpy as np

def get_ax(rows=1, cols=1, size=12, shape=None):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    """
    if shape is None:
        shape=(size*cols, size*rows)
    else:
        shape=(shape[0]*size, shape[1]*size)
    _, ax = plt.subplots(rows, cols, figsize=shape)
    return ax

def get_bbox_picture(x0:float, y0:float, x1:float, y1:float, shape:tuple):
    '''
    Return numpy array with rectangle.
    '''
    from skimage.draw import rectangle_perimeter
   
    bbox = np.zeros(shape)
    coords = rectangle_perimeter((y0, x0), (y1, x1), shape=shape)
    bbox[coords] = 1
    return bbox

