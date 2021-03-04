import boot
import argparse
from models.unet import UNet
from src.dsets import BroccoliDataset
import torch
import torchvision.transforms as tt
import os
import matplotlib.pyplot as plt
import datetime
import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from sklearn.metrics import confusion_matrix
import seaborn as sns
from zipfile import ZipFile 
import yaml
import io
import sys
def diceLoss(prediction_g: torch.tensor, label_g: torch.tensor, epsilon: int =1) -> int:
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

def iou(target_mask, prediction_mask):
    """
    Function to compute Iou given target_mask and prediction_mask
    Args:
        target_mask ([torch.tensor]): [torch.tensor describing the ground truth mask]
        prediction_mask ([torch.tensor]): [torch.tensor describing the predicted mask]

    Returns:
        [integer]: [Iou computed]
    """
    target_mask = target_mask.squeeze().cpu().numpy()
    prediction_mask = prediction_mask.squeeze().cpu().numpy()
    intersection = np.logical_and(target_mask, prediction_mask)
    union = np.logical_or(target_mask, prediction_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def main(dataset_images_path: str, dataset_masks_path: str, model_path: str, number_masks_visualize: int=0):
    """
    Main function to ininitializa dataset and model and run evaluation of the model.
    Args:
        dataset_images_path ([string]): [Path to dataset images]
        dataset_masks_path ([string]): [Path to dataset masks]
        model_path ([string]): [Path to model]
        number_masks_visualize ([integer]): [Number of predicted masks to visualize]
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    test_ds = BroccoliDataset(image_dir=dataset_images_path, mask_dir=dataset_masks_path, transform=tt.Compose([tt.ToTensor()]))
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=use_cuda)
    with ZipFile(model_path, 'r') as zip_file:
        model_conf = zip_file.read("conf.yml")
        model_dict = io.BytesIO(zip_file.read("model.pth"))
        model_conf = yaml.safe_load(model_conf)
        segmentation_model = UNet(**model_conf)    
        segmentation_model = segmentation_model.to(device)
        segmentation_model.load_state_dict(torch.load(model_dict))   
        
    dice_loss_tot = 0
    iou_loss_tot = 0
    num = 0
    list_labels = []
    list_preds = []
    # list_idx_plots = []
    with torch.no_grad():
      for idx, batch in enumerate(test_dl):
        
        image , label, id_filename = batch
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        segmentation_model.eval()
        pre = segmentation_model(image).round()
        aux_label = 0
        aux_pre = 0
        if torch.sum(pre) > 5:
            aux_pre = 1
            list_preds.append(1)
        else:
            aux_pre = 0
            list_preds.append(0)
             

        if "empty_box" in id_filename[0]:
            aux_label = 0
            list_labels.append(0)
        else:
            aux_label = 1
            list_labels.append(1)

        # if aux_label==1 and aux_pre==0:
        #     list_idx_plots.append(idx)
        
        loss = round(diceLoss(pre,label).item(), 3)
        dice_loss_tot+=loss
        iou_loss_tot+= iou(label, pre)
        image = image.to('cpu')
        label = label.to('cpu')
        pre = pre.to('cpu')
        pre = pre[0].detach().numpy()[0]
        image = image.squeeze().permute(1,2,0)
        label = label.squeeze()
        if number_masks_visualize > test_ds.__len__() :
            raise Exception("The number of masks to visualize is grather than the number of test masks")
        if num < number_masks_visualize:
            visualize_evaluation(image, label, pre, loss)
            num+=1

    # for idx in list_idx_plots:
    #     images, labels, _ = test_ds[idx]
    #     images = images.to(device, non_blocking=True)
    #     label = label.to(device, non_blocking=True)
    #     segmentation_model.eval()
    #     pre = segmentation_model(images.unsqueeze(0))
    #     fig = plt.figure()
    #     plt.imshow(images.permute(1,2,0).to('cpu'))

    #     # plt.imshow(pre[0].round().detach().permute(1,2,0).to('cpu'))
    #     fig.savefig('plot'+str(idx)+'.png')

    print("Total number samples", test_ds.__len__() )
    print("Mean Dice Loss : ",dice_loss_tot/test_ds.__len__() )
    print("Mean Iou: ", iou_loss_tot/test_ds.__len__())

    cf_matrix = confusion_matrix(list_labels, list_preds)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    cf_matrix_text = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    cf_matrix_text = np.asarray(cf_matrix_text).reshape(2,2)
    fig = sns.heatmap(cf_matrix, annot=cf_matrix_text, fmt="", cmap='coolwarm', linewidths=1).set_title("Unet model confusion matrix")
    fig.get_figure().savefig("Unet_cf_matrix.png")

def visualize_evaluation(image: torch.tensor, label: torch.tensor, pre: torch.tensor, loss: int):
    """
    Function to visulize preditictions
    Args:
        image (torch.tensor): [Real image]
        label (torch.tensor): [Ground truth mask]
        pre ([torch.tensor]): [Predcted mask]
        loss ([integer): [Loss computed between ground truth and prediction]
    """    
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 3))
    fig.patch.set_facecolor('white')
    ax[0].title.set_text('Real Image')
    ax[0].imshow(image)

    ax[1].title.set_text('Ground Truth')
    ax[1].imshow(label)

    ax[2].title.set_text('Predicted Mask')
    ax[2].imshow(pre)


    ax[3].text(0.5, 0.5, "Dice Loss: " + str(loss), ha='center', va='center', fontsize=20)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].title.set_text("Metric")
    plt.show()

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
        help='Path to Broccoli Head Masks dataset'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model file'
    )
    
    
    parser.add_argument(
        '--number_masks_visualize',
        type=int,
        required=True,
        help='Number of masks to visualize'
    )
    args = parser.parse_args()
    main(**vars(args))