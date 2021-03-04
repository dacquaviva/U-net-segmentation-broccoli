import numpy as np
import skimage.filters as filters
import cv2
from skimage.filters import threshold_otsu
from skimage.segmentation import morphological_chan_vese
from skimage.segmentation import circle_level_set
from skimage import img_as_float

import src.colour_indices as ci
import src.image_processing as ip


def unsupervised_thresholding_alg(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding
    text_threshold = filters.threshold_local(image, block_size=51, offset=10)
    image = image > text_threshold
    image = image.astype(np.float32)

    # Apply Circle Mask
    height, width = image.shape
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (width//2, height//2), width//2, 255, thickness=-1)
    image = cv2.bitwise_and(image, image, mask=circle_img)

    # Define the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = image.astype(np.float32)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def otsu_thresholding_alg(image):
    # image = cv2.GaussianBlur(image,(0,0),3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    val = threshold_otsu(image)
    image = image < val
    image = image.astype(np.float32)
    image = np.abs(image - 1)  # switch mask assignining 1 to broccoli
    return image


def morphological_chan_vese_alg(image):
    # leaf mask
    # a = otsu_thresholding_alg(image)
    image = np.uint8(ip.Scale(ci.BGR2COM1(image), feature_wise=False)*255)
    image = img_as_float(image)
    image = cv2.GaussianBlur(image, (3, 3), 3)
    height, width = image.shape
    radius = max(height, width)

    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    mask_broccoli = morphological_chan_vese(image,
                                            iterations=10,
                                            init_level_set=circle_level_set(
                                                image.shape, (height//2, width//2), radius//2),
                                            smoothing=4,
                                            lambda1=2, lambda2=1, iter_callback=callback)

    mask_broccoli = mask_broccoli.astype(np.uint8)
    return mask_broccoli, evolution


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def grabcut_alg(image, mask):

    height, width = mask.shape
    radius = max(height, width)

    mask[mask > 0] = cv2.GC_PR_FGD
    mask[mask == 0] = cv2.GC_BGD
    mask = cv2.circle(mask, (height//2, width//2),
                      radius//3, cv2.GC_FGD, thickness=-1)
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    # apply GrabCut using the the mask segmentation method

    image = cv2.GaussianBlur(image, (3, 3), 3)

    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, None, bgModel,
                                           fgModel, iterCount=100, mode=cv2.GC_INIT_WITH_MASK)

    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                          0, 1)
    
    outputMask = outputMask.astype(np.uint8)

    return outputMask
