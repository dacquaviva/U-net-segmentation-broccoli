import json
import boot
import argparse
from src import utils as utils
from src import segmentation_algs as segmentation
from src import metrics as metrics
from src import visualization as plot
import cv2
import os

from tqdm import tqdm



def main(dataset_path, number_masks_visualize):
    """
    Main function to run pipeline to evaluate unsupevised approach
    Args:
        dataset_path ([string]): [Path to dataset images]
        number_masks_visualize ([integer]): [Number of predicted masks to visualize]
    """    
    annotations = json.load(open(os.path.join(dataset_path, "mask_annotations.json")))
    sum_metrics = {}
    num = 0
    for k, a in tqdm(annotations.items()):
    
        image_path, image, polygon = utils.get_image_property(a, dataset_path)
        
        target_mask = utils.create_numpy_from_mask(image, polygon)

        evolution = None

        prediction_mask, evolution = segmentation.morphological_chan_vese_alg(image)
        
        prediction_mask = segmentation.grabcut_alg(image, prediction_mask)
        
        #Debug
        # x, y, w, h = cv2.boundingRect(prediction_mask)
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)

        validation_metrics = metrics.get_validation_metrics(target_mask, prediction_mask)

        sum_metrics = {k: validation_metrics.get(k, 0) + sum_metrics.get(k, 0) for k in set(validation_metrics)}

        # To visulise the snake evolution
        # plot.plot_evaluation(image_path, image, prediction_mask, target_mask, validation_metrics, evolution)
        if number_masks_visualize > len(annotations):
            raise Exception("The number of masks to visualize is grather than the number of test masks")
        
        if num < number_masks_visualize:
            plot.plot_evaluation(image_path, image, prediction_mask, target_mask, validation_metrics)
            num+=1
        
    num_test_sample = len(annotations)
    mean_metrics = {k: v/num_test_sample for k, v in sum_metrics.items()}
    print("Mean Metrics = ", mean_metrics)
    print("Number of test data : ", num_test_sample)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Broccoli Head Segmentation')
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to Broccoli Head dataset'
    )
    
    parser.add_argument(
        '--number_masks_visualize',
        type=int,
        required=True,
        help='Number of masks to visualize'
    )
    args = parser.parse_args()
    main(**vars(args))