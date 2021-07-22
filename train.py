!pip install segmentation-models-pytorch
!pip install pytorch-lightning
!pip install neptune-client

import segmentation_models_pytorch as smp
import json
import os
import sys
import random
import logging
import cv2
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from pytorch_lightning import Callback
from itertools import chain
import logging
from pytorch_lightning.loggers import NeptuneLogger
import re
import time
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pytesseract
import neptune.new as neptune
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def mask_label(js,v) -> 'masked image.shape= (3,1024,800)' :
    """
    
    Args:
        js: json = input labels 
        v : str = <image_name>.jpg
    returns:
        3 channel mask for each class as values of the dicitonary mask/
            eg.arr[0]-> a numpy array with filled rectangular boxes at the guided locations of text.
            similarly for arr[1] and arr[2] for bar_code and qr_code respectively.
        ->arr[0].shape = arr[1] = arr[2] = [1024,800]
        
    """
    arr = np.full((3,1024,800),0,dtype=np.float32)    
    for i in js[v] :    
        if i['type']=='text':        
            cv2.rectangle(arr[0],(i['geometry'][0][0],i['geometry'][0][1]),(i['geometry'][1][0],i['geometry'][1][1]), (255,255,255), -1)
        elif i['type']=='bar_code':
            cv2.rectangle(arr[1],(i['geometry'][0][0],i['geometry'][0][1]),(i['geometry'][1][0],i['geometry'][1][1]), (255,255,255), -1)
        else:
            cv2.rectangle(arr[2],(i['geometry'][0][0],i['geometry'][0][1]),(i['geometry'][1][0],i['geometry'][1][1]), (255,255,255), -1)    
    return np.where(arr==255,1,0)
  
  
  
  class img_dataset(Dataset):
    def __init__(self,path:str,label:str,val=False):
        """
        Args:
            path : location of the image folder
            label = If True, is the location of the label. Expected in json format.
        
        """
        self.path = path
        if val==False:
            self.root = train_set
        else:
            self.root = val_set
        self.label = label
        with open(self.label) as f:
            self.mask_label = json.load(f)        
        
    def __len__(self):
        return len(self.root)    
    def __getitem__(self,idx):
        image = cv2.imread(os.path.join(self.path,self.root[idx]))
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        th3=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,3)
        ret, threshold = cv2.threshold(th3,0,255,cv2.THRESH_OTSU)        
        threshold = threshold[np.newaxis,:]               
        masked_labels = mask_label(self.mask_label,self.root[idx])
        return {"image":torch.tensor(threshold,dtype = torch.float32),"label":torch.tensor(masked_labels,dtype=torch.float32)}
  
  
  class DiceBCELoss(nn.Module):
    """Args:
    inputs : output from the segmentation head on top. 
    inputs.shape = [3,1024,800]
    target : masked_images of 3 channels as labels.    
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):       
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        inter = (inputs * targets).sum()                            
        dice_loss = 1 - (2.0*inter + 1)/(inputs.sum() + targets.sum() +1)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE =BCE + dice_loss        
        return Dice_BCE
    
def dice_metrics(ar1,ar2):     
    ar1 = F.sigmoid(ar1)    
    co = (ar1*ar2).sum()
    return (2*co)/(torch.sum(ar1+ar2)+1e-8)
  
 ###########################################################################################################################################################################

#Training

 class train_unet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model        
        self.criterion = DiceBCELoss()       
        
    def forward(self,x):        
        return self.model(x)
    
    def training_step(self,batch,batch_idx):
        x,y = batch["image"],batch["label"]
        output = self(x)       
        loss = self.criterion(output,y)        
        self.logger.experiment.log_metric('train_loss_step',loss)
        self.logger.experiment.log_metric('train_dice_metric',dice_metrics(output,y,eps=1e-8))
        
        return loss
    
    def training_epoch_end(self,outputs):                  
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.log_metric('train_loss_epoch',avg_train_loss)
    
    def validation_step(self,batch,batch_idx):
        x,y = batch["image"],batch["label"]
        output = self(x)       
        val_loss = self.criterion(output,y)        
        self.logger.experiment.log_metric('val_loss_step',val_loss)
        self.logger.experiment.log_metric('val_dice_metric',dice_metrics(output,y,eps=1e-8))
        self.log('val_loss',val_loss)
        return val_loss
    
    def validation_epoch_end(self,validation_step_outputs):
        av_loss = torch.stack([x for x in validation_step_outputs]).mean()  
        self.logger.experiment.log_metric('val_loss_epoch',av_loss)
        
        
    def train_dataloader(self):
        train_dataset = img_dataset(path = "../input/zipfolder",label="../input/labelimages/labels-a0349fd8.json",val=False)
        return DataLoader(train_dataset,batch_size=2,num_workers=4)
    
    def val_dataloader(self):
        val_dataset = img_dataset(path = "../input/zipfolder",label="../input/labelimages/labels-a0349fd8.json",val=True)
        return DataLoader(val_dataset,batch_size=1,num_workers = 4) 
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 0.0001) 
      
trainer = pl.Trainer(gpus=-1,max_epochs =5,logger=neptune_logger,check_val_every_n_epoch=1)
modell = train_unet()
trainer.fit(modell)
trainer.save_checkpoint("cls_res50.ckpt")
