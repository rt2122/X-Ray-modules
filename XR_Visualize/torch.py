import torch
import numpy as np
from skimage import measure
from XR_Visualize import get_ax, get_bbox_picture
from matplotlib import patches
from skimage.filters import gaussian
import cv2


def get_contour(mask: np.ndarray, thr: float = 0.8):
    mask[np.where(mask >= thr)] = 1
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_info = []
    for c in contours:
        d = {'contour': c, 'area': cv2.contourArea(c)}
        contour_info.append((d))
    contour_info = sorted(contour_info, key=lambda c: c['area'], reverse=True)
    contour = contour_info[0]['contour']
    return contour[:, 0, :]


def show_prediction(pic: torch.Tensor, pred: dict, coef: float = 0.3,
                    box_c: str = 'r', mask_c: str = 'g', title: str = '', limits: tuple = None,
                    mask_mode: bool = False, alpha: float = 0.5, gauss_coef: int = None):
    '''
    Visualize input picture and predictions from pytorch Mask-RCNN
    '''
    # Check order of channels
    ax = get_ax()
    pic = pic.permute(1, 2, 0).numpy().copy()
    if gauss_coef is not None:
        pic = gaussian(pic, gauss_coef)
    ax.imshow(pic)

    # Get bboxes
    bbox = np.zeros(pic.shape[:2] + (1,))
    for i in range(len(pred['boxes'])):
        box = pred['boxes'][i].detach().numpy()
        sizes = box[2:] - box[:2]
        rect = patches.Rectangle(box[:2], *sizes, linewidth=1, edgecolor=box_c, facecolor='none')
        ax.add_patch(rect)

    # Get masks
    if not mask_mode:
        contours = []
        for i in range(len(pred['masks'])):
            cur_mask = pred['masks'][i].detach().numpy()
            if cur_mask.shape[0] == 1:
                cur_mask = cur_mask[0]
            contours.append(get_contour(cur_mask))
        for c in contours:
            ax.plot(c[:, 0], c[:, 1], linewidth=2, c=mask_c)
    else:
        mask = np.zeros_like(pred['masks'][0])
        for i in range(len(pred['masks'])):
            cur_mask = pred['masks'][i].detach().numpy()
            mask = np.logical_or(mask, cur_mask)
        ax.imshow(mask * alpha)
    ax.set_title(title)
    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
