import json
import boot
import argparse
from src import utils as utils
from src import segmentation_algs as segmentation
from src import metrics as metrics
from src import visualization as plot
import cv2
import os
import numpy as np
from tqdm import tqdm



def main(dataset_path, save_masks_path):
    """
    Main function to run pipeline to create masks given real images
    Args:
        dataset_path ([string]): [Path to dataset images]
        save_masks_path ([type]): [Path to where save predicted masks]
    """    
    annotations = json.load(open(os.path.join(dataset_path, "mask_annotations.json")))

    for _ , a in tqdm(annotations.items()):
    
        image_path, image, _ = utils.get_image_property(a, dataset_path)
        
        prediction_mask, _ = segmentation.morphological_chan_vese_alg(image)
        
        prediction_mask = segmentation.grabcut_alg(image, prediction_mask)
        
        if not(os.path.exists(save_masks_path)):
            # create the directory you want to save to
            os.mkdir(save_masks_path)
        sep = '.'
        id_filename = image_path.split(sep, 1)[0]
        np.savetxt(os.path.join(save_masks_path,"mask_" +id_filename + ".gz"), prediction_mask)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Broccoli Head Segmentation')
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to Broccoli Head dataset'
    )
    
    
    parser.add_argument(
        '--save_masks_path',
        type=str,
        required=True,
        help='Path to save Broccoli Head Masks'
    )
    args = parser.parse_args()
    main(**vars(args))