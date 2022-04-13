import numpy as np
from matplotlib import pyplot as plt

from skimage.filters import gaussian

from eR_Mask_RCNN.tf import eRDataset
from XR_Visualize import get_ax 

def show_image_with_masks(dataset:eRDataset, image_id:int, gauss_coef:float=1, idx=None):
    image = dataset.load_image(image_id).astype(np.float64)
    mask, class_ids = dataset.load_mask(image_id)
    
    mask_by_class = np.zeros((dataset.size, dataset.size, dataset.num_classes-1), dtype=np.float64)
    for i in range(1, dataset.num_classes):
        mask_by_class[:,:,i-1] = np.sum(mask[:,:,class_ids==i], axis=2)
        
    if not idx is None:
        image = image[idx]
        mask_by_class = mask_by_class[idx]
        
    image = gaussian(image, gauss_coef)
    image -= image.min()
    image /= image.max()
    image *= 255
    
    mask_by_class = np.clip(mask_by_class, 0, 1)
    mask_by_class *= 255 * 0.8
    ax = get_ax(1, 1)
   
    ax.imshow(np.dstack([image, mask_by_class[:,:,:2]]).astype(np.int32))
    print("image: min={}, max={}".format(image.min(), image.max()))
    print(mask_by_class.min(), mask_by_class.max())
    #ax.imshow(image.astype(np.int8))
