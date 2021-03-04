import boot
import argparse
from models.resnet9 import ResNet9
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
from torch import nn
from sklearn.metrics import confusion_matrix
from zipfile import ZipFile 
import yaml
import io
import sys
def  main(dataset_images_path: str, model_path: str):
    """
    Main function to ininitializa dataset and model and run evaluation of the model.
    Args:
        dataset_images_path ([string]): [Path to dataset images]
        model_path ([string]): [Path to model]
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_ds = BroccoliDataset(image_dir=dataset_images_path, transform=tt.Compose([tt.ToTensor()]))
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=use_cuda)
    
    with ZipFile(model_path, 'r') as zip_file:
        model_conf = zip_file.read("conf.yml")
        model_dict = io.BytesIO(zip_file.read("model.pth"))
        model_conf = yaml.safe_load(model_conf)
        classification_model = ResNet9(**model_conf)
        classification_model = classification_model.to(device)
        classification_model.load_state_dict(torch.load(model_dict))

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    loss_tot = 0
    acc_tot = 0
    # list_idx_plots = []
    list_labels = []
    list_preds = []
    with torch.no_grad():
        for idx,batch in enumerate(test_dl):
            images, labels  = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).reshape((labels.shape[0]))
            
            classification_model.eval()
            predictions = classification_model(images)
            # prob_dist = softmax(predictions).squeeze().tolist()

            loss = criterion(predictions, labels)
            loss_tot +=loss
            _, preds = torch.max(predictions, dim=1)
            softmax(predictions)
            acc = torch.tensor(torch.sum(preds == labels).item())
            acc_tot += acc

            # if labels==0 and preds==1:
            # # if np.abs(prob_dist[0] - prob_dist[1]) < 0.5 and labels==1 and preds==0:
            #     list_idx_plots.append(idx)

            list_labels.append(labels.to('cpu').item())
            list_preds.append(preds.to('cpu').item())
           

      



    # for idx in list_idx_plots:
    #     images, labels = test_ds[idx]
    #     fig = plt.figure()
    #     plt.imshow(images.permute(1,2,0).to('cpu'))
    #     fig.savefig('str(idx)+'.png')


    cf_matrix = confusion_matrix(list_labels, list_preds)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    cf_matrix_text = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    cf_matrix_text = np.asarray(cf_matrix_text).reshape(2,2)
    fig = sns.heatmap(cf_matrix, annot=cf_matrix_text, fmt="", cmap='coolwarm', linewidths=1).set_title("Binary classifier confusion matrix")
    fig.get_figure().savefig("classifier_cf_matrix.png")
    print("Total number samples", test_ds.__len__())
    print("Mean Loss : ",loss_tot.item()/len(test_ds))
    print("Mean Accuracy : ",acc_tot.item()/len(test_ds))



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
    
    
    args = parser.parse_args()
    main(**vars(args))