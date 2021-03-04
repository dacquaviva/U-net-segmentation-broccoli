import boot
import argparse
from models.unet import UNet
from src.dsets import BroccoliDataset
import torch
import torchvision.transforms as tt
import os
import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from zipfile import ZipFile 
import yaml
import io
import sys
def main(dataset_images_path: str, model_path: str, save_masks_path: str):
    """
    Main function to inizialize dataset and model to create masks given real images
    Args:
        dataset_images_path ([string]): [Path to dataset images]
        model_path ([string]): [Path to model]
        save_masks_path ([string]): [Path to where save predicted masks]
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_ds = BroccoliDataset(image_dir=dataset_images_path, transform=tt.Compose([tt.ToTensor()]))
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=use_cuda)

    with ZipFile(model_path, 'r') as zip_file:
        model_conf = zip_file.read("conf.yml")
        model_dict = io.BytesIO(zip_file.read("model.pth"))
        model_conf = yaml.safe_load(model_conf)
        segmentation_model = UNet(**model_conf)    
        segmentation_model = segmentation_model.to(device)
        segmentation_model.load_state_dict(torch.load(model_dict))

    with torch.no_grad():
      for batch in test_dl:
        image, id_filename = batch
        image = image.to(device, non_blocking=True)
        segmentation_model.eval()
        pre = segmentation_model(image)
        pre = pre.to('cpu')
        pre = pre[0].detach().numpy()[0]
        if not(os.path.exists(save_masks_path)):
            # create the directory you want to save to
            os.mkdir(save_masks_path)
            
        np.savetxt(os.path.join(save_masks_path, id_filename[0] + "_mask.gz"), pre)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Broccoli Head Segmentation')
    
    parser.add_argument(
        '--dataset_images_path',
        type=str,
        required=True,
        help='Path to Broccoli Head Images dataset'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model file'
    )
    
    
    parser.add_argument(
        '--save_masks_path',
        type=str,
        required=True,
        help='Path to save Broccoli Head Masks'
    )
    args = parser.parse_args()
    main(**vars(args))