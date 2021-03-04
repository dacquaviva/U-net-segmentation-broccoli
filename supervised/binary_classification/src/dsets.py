import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tt
import torchvision
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
) 
from PIL import Image

class BroccoliDataset(Dataset):
    """
    Class to handle dataset
    Args:
        Dataset ([torch.utils.data.DataLoader]): [Dataset contaning data]
    """    
    def __init__(self, image_dir: str, mask_dir: str =None, transform: torchvision.transforms =None):
        """
        Init function to inintialized class
        Args:
            image_dir (string): [Path to imges]
            mask_dir ([string], optional): [Path to masks]. Defaults to None.
            transform ([Compose], optional): [object containing transormations to applay to data]. Defaults to None.
        """        
        
        self.transform = transform
        self.list_image_id = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        
        if not (self.list_image_id):
            raise Exception("Image path path not found")

    def __len__(self) -> int:
        """
        Function to compute dataset lenght 
        Returns:
            [integer]: [Dataset lenght]
        """        
        return len(self.list_image_id)

    def __getitem__(self, index: int) -> tuple:
        """
        Function to retrieve a sample given an index
        Args:
            index ([integer]): [Index associated with sample]

        Returns:
            [tuple]: [Item containing image and relative mask]
        """        
            
        image = Image.open(self.list_image_id[index])
        image = self.transform(image)
        
        id_image_name = os.path.basename(self.list_image_id[index])
        sep = '.'
        id_filename = id_image_name.split(sep, 1)[0]
        
        if "empty_box" in id_filename:
            label = torch.tensor([[0.]], dtype=torch.long)#No Broccoli
        else:
            label  = torch.tensor([[1.]],  dtype=torch.long)#Broccoli present
        batch = (image, label)
        return batch


