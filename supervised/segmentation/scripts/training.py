import boot
import argparse
from models.unet import UNet
from src.dsets import BroccoliDataset
import torch
from torch import nn
from torch.optim import SGD, Adam
import torchvision.transforms as tt
import os
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
) 
from typing import Tuple
class SegmentationTrainingApp:
    """
    class wrapping the entire training segmentation pipeline
    """    
    def __init__(self, sys_argv=None):
    
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.segmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()
        #tensorboard
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.trn_writer,  self.val_writer  = self.initTensorboardWriters()
        self.train_dl = None
        self.val_dl = None
            
                       
    def initModel(self) -> UNet:
        """
        Function to initializa model
        Returns:
            [UNet]: [Initialized model]
        """                
        segmentation_model = UNet(            
            in_channels=3,
            n_classes=1,
            depth=5,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode='upconv',)

        if self.use_cuda:
            print("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
            segmentation_model = segmentation_model.to(self.device)
        return segmentation_model
    
    def initTensorboardWriters(self) -> Tuple[SummaryWriter, SummaryWriter]:
        """
        Function to initialize Tensorboard
        Returns:
            [SummaryWriter]: [Object to log data to Tensorboard]
        """        
        log_dir = os.path.join('runs', self.time_str) 
        self.train_writer = SummaryWriter(
        log_dir=log_dir + '_trn_seg')
        self.validation_writer = SummaryWriter(
        log_dir=log_dir + '_val_seg')
        return self.train_writer, self.validation_writer


    def main(self, dataset_images_path: str, dataset_masks_path: str, save_model_path: str):
        """
        Main function to ininitializa dataset and run traninig of the model
        Args:
            dataset_images_path ([string]): [Path to dataset images]
            dataset_masks_path ([string]): [Path to dataset masks]
            save_model_path ([string]): [Path to where save the model]
        """        

        dataset = BroccoliDataset(image_dir=dataset_images_path, mask_dir=dataset_masks_path, transform=tt.Compose([tt.ToTensor()]))

        total_sample = dataset.__len__()
        
        val_size = int(total_sample * 0.2)
        train_size = total_sample - val_size
        
    
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        batch_size = 32
        
        
        self.train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=self.use_cuda)
        self.val_dl = DataLoader(dataset=val_ds, batch_size=batch_size*2, shuffle=True, num_workers=8, pin_memory=self.use_cuda)
        
        history_loss = self.fit_model(save_model_path)
        
        
        self.trn_writer.close()
        self.val_writer.close()
        
              
    def fit_model(self, save_model_path: str) -> dict:
        """
        Function to train the model
        Args:
            save_model_path ([string): [Path to where save the model]

        Returns:
            [dict]: [Dictonary containing all metrics during training]
        """        
        history_loss = {'train':0,
                        'validation':0}
        best_validation_loss = 1000
        for epoch in range(1000):
            # Training Phase 
            self.segmentation_model.train()
            train_losses = []
            for batch in tqdm(self.train_dl):
                loss = self.computeBatchLoss(batch)
                train_losses.append(loss.detach())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            #Train
            # history_loss['train'].append(torch.stack(train_losses).mean().item())
            history_loss['train'] = torch.stack(train_losses).mean().item()
            
            # Validation
            validation_losses = self.evaluate_model()
            # history_loss['validation'].append(torch.stack(validation_losses).mean().item())
            history_loss['validation'] = torch.stack(validation_losses).mean().item()
            
            self.log_metrics_tensorboard(history_loss, epoch)
            
            # if epoch < 50:
            #     self.logImages('train', epoch)
            #     self.logImages('validation', epoch)
            
            if history_loss['validation'] < best_validation_loss:
                best_validation_loss = history_loss['validation']
                self.save_model(best_validation_loss,save_model_path)
                
            
            self.epoch_end(epoch, history_loss)
        return history_loss
    
    # def logImages(self, mode_str, epoch):
    #     """
    #     Function to log predicted masks to tensorboard
    #     Args:
    #         mode_str (string): [String specifing ]
    #         epoch ([type]): [Number of current epoch]
    #     """        
    #     if mode_str == 'train':
    #         num_images = self.train_dl.dataset.__len__()
    #     else:
    #         num_images = self.val_dl.dataset.__len__()
        
    #     
    #     num_images = 8
        
        # with torch.no_grad():
        #     for i in range(num_images):
        #         self.segmentation_model.eval()
        #         writer = getattr(self, mode_str + '_writer')
        #         if mode_str == 'train':
        #             image, mask, _ = self.train_dl.dataset.__getitem__(i)
        #         else:
        #             image, mask, _ = self.val_dl.dataset.__getitem__(i)
        #         image = image.to(self.device, non_blocking=True)
        #         pre = self.segmentation_model(image.unsqueeze(0))
        #         writer.add_image(f'{mode_str}/{i}_prediction',pre.reshape(1,64,64),epoch,dataformats='CHW')
                
        #         if epoch == 0:
        #             writer.add_image(f'{mode_str}/{i}_label',image,epoch, dataformats='CHW')
        #         writer.flush()


    def log_metrics_tensorboard(self, history: dict, epoch: int):
        """
        Function to log metrics to tensorbard
        Args:
            history (dict): [Dictonary containing all metrics during training]
            epoch ([integer]): [Number of current epoch]
        """        
        
        for key, value in history.items():
            writer = getattr(self, key + '_writer')
            writer.add_scalar(key, value, epoch)

        writer.flush()
    
    
    def epoch_end(self, epoch: int, history: dict):
        """
        Function to print status of training at each epoch
        Args:
            epoch ([integer]): [Number of current epoch]
            history ([dict]): [Dictonary containing all metrics during training]
        """        
        print("Epoch [{}], Dice Loss Train: {:.5f}, Dice Loss Val: {:.5f}".format(epoch, history['train'], history['validation']))

    
    
    def evaluate_model(self) -> int:
        """
        Function to evalute the model on validation set
        Returns:
            [integer]: [Loss computed on validation set]
        """        
        with torch.no_grad():
            self.segmentation_model.eval()
            outputs = [self.computeBatchLoss(batch).detach() for batch in self.val_dl]
        return outputs

    def computeBatchLoss(self, batch: torch.tensor) -> int:
        """
        Function to compute the loss across an entire batch
        Args:
            batch ([torch.tensor]): [Batch of data]

        Returns:
            [integer]: [loss computed across an entire batch]
        """        
        image, mask, _ = batch
        
        image = image.to(self.device, non_blocking=True)
        real_mask = mask.to(self.device, non_blocking=True)
        
        predicted_mask = self.segmentation_model(image)
        
        diceLoss_g = self.diceLoss(predicted_mask, real_mask)

        return diceLoss_g.mean()


    def diceLoss(self, prediction_g: torch.tensor, label_g: torch.tensor, epsilon:int =1) -> int:
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
    
    


    def initOptimizer(self) -> torch.optim:
        """
        Initialization optimazier
        Returns:
            [torch.optim]: [Optimazer of the model]
        """              
        return Adam(self.segmentation_model.parameters())
        # return SGD(self.segmentation_model.parameters(), lr=0.001, momentum=0.99)
        


    def save_model(self, loss: int, save_model_path: str):
        """
        Save weights of the model
        Args:
            loss ([integer]): Loss associated with the model configuation.
            save_model_path ([string]): Path location where to save weights model.
        """        
        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        if not(os.path.exists(save_model_path)):
            # create the directory you want to save to
            os.mkdir(save_model_path)
        torch.save(model.state_dict(), os.path.join(save_model_path, 'segmentation_model.pth'))
        print("save model loss: ", loss)


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
        '--save_model_path',
        type=str,
        required=True,
        help='Path to save the best model'
    )
    args = parser.parse_args()
    SegmentationTrainingApp().main(**vars(args))