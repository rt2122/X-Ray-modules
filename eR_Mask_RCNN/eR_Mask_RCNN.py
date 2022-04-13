import pandas as pd
import numpy as np
from skimage.draw import disk 

def cat2masks(path: str, raw_size: int = 3240, grid_size: int = 270) -> tuple:
    """
    Read csv file with coordinates of sources and transform it to masks and labels.
    """    
    size = raw_size - 2 * grid_size
    cat = pd.read_csv(path)
    #Remove sources from grid
    for coord in 'xy':
        cat[coord] -= grid_size
        cat = cat[cat[coord] >= 0]
        cat = cat[cat[coord] < size]
    
    sources = len(cat)
    masks = np.zeros((size, size, sources), dtype=np.uint8)
    labels = np.zeros(sources)
    labels[cat['EXT'] > 0] = 1 #Extended sourses
    labels[cat['EXT'] == 0] = 2 #Point sources
    
    for i in range(sources):
        line = cat.iloc[i]
        coords = disk((line['x'], line['y']), line['pix_rad'], shape=(size, size, 1))
        masks[:,:,i][coords] = 1
    return masks, labels
