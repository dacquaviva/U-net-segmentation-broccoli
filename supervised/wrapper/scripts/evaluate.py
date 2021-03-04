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
from zipfile import ZipFile 
import yaml
import io
import sys
def diceLoss(prediction_g: torch.tensor, label_g: torch.tensor, epsilon: int=1) -> int:
    """
    Function to compute the Dice loss between prediction_g and label_g
    Args:
        prediction_g ([torch.tensor]): [torch.tensor describing the predicted mask]
        label_g ([torch.tensor]): [torch.tensor describing the ground truth mask]
        epsilon ([int, optional]): [Avoid to divide by 0]. Defaults to 1.

    Returns:
        [integer]: [Dice loss computed]
    """ 
    diceLabel_g = label_g.sum(dim=[1,2,3])
    dicePrediction_g = prediction_g.sum(dim=[1,2,3])
    diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

    diceRatio_g = (2 * diceCorrect_g + epsilon) \
        / (dicePrediction_g + diceLabel_g + epsilon)

    return 1 - diceRatio_g

def  main(dataset_images_path, dataset_masks_path: str, classification_model_path: str, segmentation_model_path: str):
    """
    Main function to ininitializa dataset and model and run evaluation of the model.
    Args:
        dataset_images_path ([string]): [Path to dataset images]
        dataset_masks_path ([string]): [Path to dataset masks]
        classification_model_path ([string]): [Path to classifier model]
        segmentation_model_path ([string]): [Path to segmentation model]
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_ds = BroccoliDataset(image_dir=dataset_images_path, mask_dir=dataset_masks_path, transform=tt.Compose([tt.ToTensor()]))
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=use_cuda)

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
    dice_loss_tot = 0

    with torch.no_grad():
      for idx, batch in enumerate(test_dl):
        
        image , label, id_filename = batch
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        pre = wrapper_model(image).round().to(device, non_blocking=True)    
        loss = round(diceLoss(pre,label).item(), 3)
        dice_loss_tot+=loss
        
    


    print("Total number samples", test_ds.__len__() )
    print("Mean Dice Loss : ",dice_loss_tot/test_ds.__len__() )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Broccoli Head Segmentation')
    
    parser.add_argument(
        '--dataset_images_path',
        type=str,
        required=True,
        help='Path to Broccoli Head Images dataset'
    )

    parser.add_argument(
        '--dataset_masks_path',
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