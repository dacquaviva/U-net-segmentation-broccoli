import cv2
import numpy as np
import skimage
import os


def get_image_property(annotation, dataset_path):
    image_path = annotation['filename']
    image = load_image(os.path.join(dataset_path, image_path)) 
    polygon = annotation['regions'][0]['shape_attributes']
    return image_path, image, polygon


def load_image(image_path):
    image = cv2.imread(image_path)
    return image


def create_numpy_from_mask(image, polygon):
    height, width = image.shape[:2]
    target_mask = np.zeros([height, width], dtype=np.uint8)
    rr, cc = skimage.draw.polygon(
        polygon['all_points_y'], polygon['all_points_x'])
    target_mask[rr, cc] = 1
    return target_mask
