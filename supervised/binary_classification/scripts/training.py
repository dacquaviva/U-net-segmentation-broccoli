import boot
import argparse
from models.resnet9 import ResNet9
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
class BinaryClassificationTrainingApp:
    """
    class wrapping the entire training binary classification pipeline
    """    
    def __init__(self, sys_argv=None):
    
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.classification_model = self.initModel()
        self.optimizer = self.initOptimizer()
        self.criterion = self.initCriterion()
        #tensorboard
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.train_loss_writer, self.validation_loss_writer,  self.validation_accuracy_writer = self.initTensorboardWriters()
        self.train_dl = None
        self.val_dl = None
            
                       
    def initModel(self) -> ResNet9:
        """
        Function to initializa model
        Returns:
            [ResNet9]: [Initialized model]
        """                
        classification_model = ResNet9(            
            in_channels=3,)

        if self.use_cuda:
            print("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                classification_model = nn.DataParallel(classification_model)
            classification_model = classification_model.to(self.device)
        return classification_model
    
    def initTensorboardWriters(self) -> Tuple[SummaryWriter, SummaryWriter, SummaryWriter]:
        """
        Function to initialize Tensorboard
        Returns:
            [SummaryWriter]: [Object to log data to Tensorboard]
        """        
        log_dir = os.path.join('runs', self.time_str) 
        self.train_loss_writer = SummaryWriter(
        log_dir=log_dir + '_trn_classificaiton')

        self.validation_loss_writer = SummaryWriter(
        log_dir=log_dir + '_val_loss_classification')

        self.validation_accuracy_writer = SummaryWriter(
        log_dir=log_dir + '_val_acc_classificatin')
        return self.train_loss_writer, self.validation_loss_writer,  self.validation_accuracy_writer


    def main(self, dataset_images_path: str, save_model_path: str):
        """
        Main function to ininitializa dataset and run traninig of the model
        Args:
            dataset_images_path ([string]): [Path to dataset images]
            save_model_path ([string]): [Path to where save the model]
        """        

        dataset = BroccoliDataset(image_dir=dataset_images_path, transform=tt.Compose([tt.ToTensor()]))

        total_sample = dataset.__len__()
        
        val_size = int(total_sample * 0.2)
        train_size = total_sample - val_size
        
    
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        batch_size = 32
        
        
        self.train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=self.use_cuda)
        self.val_dl = DataLoader(dataset=val_ds, batch_size=batch_size*2, shuffle=True, num_workers=8, pin_memory=self.use_cuda)
        
        history_loss = self.fit_model(save_model_path)
        
        self.train_loss_writer.close()
        self.validation_loss_writer.close()
        self.validation_accuracy_writer.close() 
        
              
    def fit_model(self, save_model_path: str) -> dict:
        """
        Function to train the model
        Args:
            save_model_path ([string): [Path to where save the model]

        Returns:
            [dict]: [Dictonary containing all metrics during training]
        """        
        history_loss = {'train_loss':0,
                        'validation_loss': 0,
                        'validation_accuracy': 0}
        best_validation_loss = 1000
        for epoch in range(1000):
            # Training Phase 
            self.classification_model.train()
            train_losses = []
            for batch in tqdm(self.train_dl):
                loss = self.computeBatchLoss(batch)
                train_losses.append(loss.detach())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
           
            #Train
            # history_loss['train'].append(torch.stack(train_losses).mean().item())
            history_loss['train_loss'] = torch.stack(train_losses).mean().item()
            
            # Validation
            val_loss, val_acc = self.evaluate_model()
            # history_loss['validation'].append(torch.stack(validation_losses).mean().item())
            history_loss['validation_loss'] = torch.stack(val_loss).mean().item()
            history_loss['validation_accuracy'] = torch.stack(val_acc).mean().item()
            
            self.log_metrics_tensorboard(history_loss, epoch)
            
        
            
            if history_loss['validation_loss'] < best_validation_loss:
                best_validation_loss = history_loss['validation_loss']
                self.save_model(best_validation_loss,save_model_path)
                
            
            self.epoch_end(epoch, history_loss)
        return history_loss
    

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
        print("Epoch [{}], Loss Train: {:.5f}, Loss Val: {:.5f}, Acc Val: {}".format(epoch, history['train_loss'], history['validation_loss'], history['validation_accuracy']))

    
    
    def evaluate_model(self) -> Tuple[int, int]:
        """
        Function to evalute the model on validation set
        Returns:
            [Tuple[int, int]]: [Loss computed on validation set]
        """        
        with torch.no_grad():
            self.classification_model.eval()
            val_loss = [self.computeBatchLoss(batch).detach() for batch in self.val_dl]
            val_acc = [self.computeBatchAcc(batch).detach() for batch in self.val_dl]

        return val_loss, val_acc

    def computeBatchLoss(self, batch: torch.tensor) -> int:
        """
        Function to compute the loss across an entire batch
        Args:
            batch ([torch.tensor]): [Batch of data]

        Returns:
            [integer]: [loss computed across an entire batch]
        """        
        images, labels = batch

        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True).reshape((labels.shape[0]))
        predictions = self.classification_model(images)
        loss = self.criterion(predictions, labels)

        return loss

    def computeBatchAcc(self, batch: torch.tensor) -> int:
        """
        Function to compute the loss across an entire batch
        Args:
            batch ([torch.tensor]): [Batch of data]

        Returns:
            [integer]: [loss computed across an entire batch]
        """        
        images, labels  = batch
        
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True).reshape((labels.shape[0]))
        
        outputs = self.classification_model(images)
        _, preds = torch.max(outputs, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))

        return acc


    def initOptimizer(self) -> torch.optim:
        """
        Initialization optimazier
        Returns:
            [torch.optim]: [Optimazer of the model]
        """              
        return Adam(self.classification_model.parameters())
        # return SGD(self.classification_model.parameters(), lr=0.001, momentum=0.99)

    def initCriterion(self) -> torch:
        """
        Initalization loss function
        Returns:
            [torch]: [Optimazer of the model]
        """
        criterion = nn.CrossEntropyLoss()
        return criterion
        


    def save_model(self, loss: int, save_model_path: str):
        """
        Save weights of the model
        Args:
            loss ([integer]): Loss associated with the model configuation.
            save_model_path ([string]): Path location where to save weights model.
        """        
        model = self.classification_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        if not(os.path.exists(save_model_path)):
            # create the directory you want to save to
            os.mkdir(save_model_path)
        torch.save(model.state_dict(), os.path.join(save_model_path, 'classifier.pth'))
        print("Save model: ", loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Broccoli Head Segmentation')
    
    parser.add_argument(
        '--dataset_images_path',
        type=str,
        required=True,
        help='Path to Broccoli Head Images dataset'
    )
    
    
    parser.add_argument(
        '--save_model_path',
        type=str,
        required=True,
        help='Path to save the best model'
    )
    args = parser.parse_args()
    BinaryClassificationTrainingApp().main(**vars(args))
