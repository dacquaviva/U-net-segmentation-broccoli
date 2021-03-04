import boot
import argparse
from models.wrapper import WrapperModel
from models.resnet9 import ResNet9
from models.unet import UNet
from src.dsets import BroccoliDataset
import torch
import torchvision.transforms as tt
import os
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import cv2
from PIL import Image
from zipfile import ZipFile 
import yaml
import io
import sys
def resize_bounding_box_size(max, start_point, size, bbox_size):
    """
    Function to compute the Dice loss between prediction_g and label_g
    Args:
        prediction_g ([torch.tensor]): [torch.tensor describing the predicted mask]
        label_g ([torch.tensor]): [torch.tensor describing the ground truth mask]
        epsilon ([int, optional]): [Avoid to divide by 0]. Defaults to 1.

    Returns:
        [integer]: [Dice loss computed]
    """ 
    mismatch = bbox_size - size
    start_point = start_point - mismatch // 2 
    size = size + mismatch
    if start_point  < 0:
        start_point = 0
        size = bbox_size 
    elif start_point+size > max:
        start_point = max - bbox_size
        size = bbox_size  
    return start_point, size

def init_device():
    """
    Function to initialize device
    """ 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def init_model(device, classification_model_path: str, segmentation_model_path: str) -> WrapperModel:
    """
    Function to initialize wrapper model
    Args:
        classification_model_path ([string]): [Path to classifier model]
        segmentation_model_path ([string]): [Path to segmentation model]

    Returns:
        [WrapperModel]: [Wrapper model]
    """ 
    with ZipFile(classification_model_path, 'r') as zip_file:
        model_conf = zip_file.read("conf.yml")
        model_dict = io.BytesIO(zip_file.read("model.pth"))
        model_conf = yaml.safe_load(model_conf)
        classification_model = ResNet9(**model_conf)
        classification_model = classification_model.to(device)
        classification_model.load_state_dict(torch.load(model_dict))

    with ZipFile(segmentation_model_path, 'r') as zip_file:
        model_conf = zip_file.read("conf.yml")
        model_dict = io.BytesIO(zip_file.read("model.pth"))
        model_conf = yaml.safe_load(model_conf)
        segmentation_model = UNet(**model_conf)    
        segmentation_model = segmentation_model.to(device)
        segmentation_model.load_state_dict(torch.load(model_dict))

    wrapper_model = WrapperModel(classification=classification_model, segmentation=segmentation_model)
    wrapper_model = wrapper_model.to(device)
    return wrapper_model



def  main(dataset_images_path, annotations_broccoli_path: str, classification_model_path: str, segmentation_model_path: str):
    """
    Main function to ininitializa dataset and model and run evaluation of the model.
    Args:
        dataset_images_path ([string]): [Path to dataset images]
        annotations_broccoli_path ([string]): [Path to dataset annotation]
        classification_model_path ([string]): [Path to classifier model]
        segmentation_model_path ([string]): [Path to segmentation model]
    """
    device = init_device()
    model = init_model(device, classification_model_path, segmentation_model_path)


    for filename in tqdm(os.listdir(annotations_broccoli_path)):
    # for filename in ["20190828_c08_biobrass_img000988_2176_1568_512_512.txt"]:
        sep = '.'
        id_filename = filename.split(sep, 1)[0]
        f = open(os.path.join(annotations_broccoli_path, id_filename + ".txt"), "r")
        tile = cv2.imread(os.path.join(dataset_images_path,id_filename + ".png"))
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        mask_tile = np.zeros([512, 512])
        for idx, row in enumerate(f):
            label, x, y, w, h = row.split()
            x = int(float(x))
            y = int(float(y))
            w = int(float(w))
            h = int(float(h))

            bbox_size = 64
            y, h = resize_bounding_box_size(tile.shape[0], y , h - y, 64)
            x, w = resize_bounding_box_size(tile.shape[1], x , w - x, 64)

            broccoli = tile[y:y+h, x:x+w]
            broccoli = Image.fromarray(broccoli)
            transform = tt.Compose([tt.ToTensor()])
            broccoli = transform(broccoli)
            broccoli = broccoli.to(device, non_blocking=True)


            pre = model(broccoli.unsqueeze(0)).round()
            

            pre = pre.to('cpu')
            pre = pre.squeeze().detach().numpy()

           
            mask_tile[y:y+h, x:x+w] = pre

        # fig = plt.figure()
        # plt.imshow(mask_tile)
        # fig.savefig('mask/' + str(id_filename)+ str(idx)+ '.png')

        # plt.imshow(tile)
        # fig.savefig('tile/' + str(id_filename)+ str(idx)+ '.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Broccoli Head Segmentation')
    
    parser.add_argument(
        '--dataset_images_path',
        type=str,
        required=True,
        help='Path to Broccoli Head Images dataset'
    )

    parser.add_argument(
        '--annotations_broccoli_path',
        type=str,
        required=True,
        help='Path to Broccoli Head Images dataset'
    )
    
    parser.add_argument(
        '--classification_model_path',
        type=str,
        required=True,
        help='Path to classification model file'
    )

    parser.add_argument(
        '--segmentation_model_path',
        type=str,
        required=True,
        help='Path to segmatation model file'
    )
    
    
    args = parser.parse_args()
    main(**vars(args))