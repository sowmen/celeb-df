import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import gc
try:
   import cPickle as pkl
except:
   import pickle as pkl

import wandb

from torch.backends import cudnn
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cudnn.benchmark = True

from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur
from transforms import IsotropicResize

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from dataset import CelebDF_Dataset
from sklearn import metrics

from utils import EarlyStopping, AverageMeter

DATA_ROOT = 'face_data'
OUTPUT_DIR = 'weights'
device = 'cuda'
config_defaults = {
    'epochs' : 30,
    'train_batch_size' : 40,
    'valid_batch_size' : 32,
    'optimizer' : 'radam',
    'learning_rate' : 1e-3,
    'weight_decay' : 0.0005,
    'schedule_patience' : 5,
    'schedule_factor' : 0.25,
    'rand_seed' : 777,
    'oversample' : True
}

def train(name, val_fold, run, folds_csv):
    
    wandb.init(project='dfdc', 
               config=config_defaults,
               name=f'{name},val_fold:{val_fold},run{run}')
    config = wandb.config
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = timm.create_model('xception', pretrained=True, num_classes=1)
    model.to(device)
    # model = DataParallel(model).to(device)
    wandb.watch(model)
    
    if config.optimizer == 'radam' :
        optimizer = torch_optimizer.RAdam(model.parameters(), 
                                          lr=config.learning_rate,
                                          weight_decay = config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                              lr=config.learning_rate,
                              weight_decay=config.weight_decay)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        threshold=0.001,
        mode="max",
        factor = config.schedule_factor
    )
    criterion = nn.BCEWithLogitsLoss()
    es = EarlyStopping(patience = 8, mode='max')
    
    data_train = CelebDF_Dataset(data_root=DATA_ROOT,
                              mode='train',
                              folds_csv=folds_csv,
                              val_fold=val_fold,
                              hardcore=True,
                              oversample_real=config.oversample,
                              transforms=create_train_transforms(size=224))
    data_train.reset(config.rand_seed)
    train_data_loader = DataLoader( data_train, 
                                    batch_size=config.train_batch_size, 
                                    num_workers=8, 
                                    shuffle=True, 
                                    drop_last=True)

    data_val = CelebDF_Dataset(data_root=DATA_ROOT,
                            mode='val',
                            folds_csv=folds_csv,
                            val_fold=val_fold,
                            hardcore=False,
                            transforms=create_val_transforms(size=224))
    data_val.reset(config.rand_seed)

    val_data_loader = DataLoader(data_val, 
                                 batch_size=config.valid_batch_size, 
                                 num_workers=8, 
                                 shuffle=False, 
                                 drop_last=True)
    

    train_history = []
    val_history = []
    
    
    for epoch in range(config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")
        
        train_metrics = train_epoch(model, train_data_loader, optimizer, criterion, epoch)
        valid_metrics = valid_epoch(model, val_data_loader, criterion, epoch)
        scheduler.step(valid_metrics['valid_auc'])

        print(f"TRAIN_AUC = {train_metrics['train_auc']}, TRAIN_LOSS = {train_metrics['train_loss']}")
        print(f"VALID_AUC = {valid_metrics['valid_auc']}, VALID_LOSS = {valid_metrics['valid_loss']}")
        
        train_history.append(train_metrics)
        val_history.append(valid_metrics)

        es(valid_metrics['valid_auc'], model, model_path=os.path.join(OUTPUT_DIR,f"{name}_fold_{val_fold}_run_{run}.h5"))
        if es.early_stop:
            print("Early stopping")
            break
    
    try:
        pkl.dump( train_history, open( f"train_history{name}{run}.pkl", "wb" ) )
        pkl.dump( val_history, open( f"val_history{name}{run}.pkl", "wb" ) )
    except:
        print("Error pickling")

    wandb.save(f'weights/{name}_fold_{val_fold}_run_{run}.h5')
      
    
    
    
def train_epoch(model, train_data_loader, optimizer, criterion, epoch):
    model.train()
    
    train_loss = AverageMeter()
    correct_predictions = []
    targets = []
    
    idx = 1
    for batch in tqdm(train_data_loader):
        
        batch_images = batch['image'].to(device)
        batch_labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        out = model(batch_images)
        
        loss = criterion(out, batch_labels.view(-1, 1).type_as(out))
        
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), train_data_loader.batch_size)
        targets.append((batch_labels.view(-1,1).cpu() >= 0.5) *1)
        correct_predictions.append(torch.sigmoid(out).cpu().detach().numpy())
        
        if(idx % 100 == 0):
            with torch.no_grad():
                temp_t = np.vstack((targets)).ravel()
                temp_correct_preds = np.vstack((correct_predictions)).ravel()

                train_auc = metrics.roc_auc_score(temp_t, temp_correct_preds)
                train_f1_05 = metrics.f1_score(temp_t,(temp_correct_preds >= 0.5)*1)
                train_acc_05 = metrics.accuracy_score(temp_t,(temp_correct_preds >= 0.5)*1)
                train_balanced_acc_05 = metrics.balanced_accuracy_score(temp_t,(temp_correct_preds >= 0.5)*1)

                train_metrics = {
                    'b_train_loss' : train_loss.avg,
                    'b_train_auc' : train_auc,
                    'b_train_f1_05' : train_f1_05,
                    'b_train_acc_05' : train_acc_05,
                    'b_train_balanced_acc_05' : train_balanced_acc_05,
                    'b_train_batch' : idx
                }
                wandb.log(train_metrics)
        idx += 1
        
    with torch.no_grad():
        targets = np.vstack((targets)).ravel()
        correct_predictions = np.vstack((correct_predictions)).ravel()

        train_auc = metrics.roc_auc_score(targets, correct_predictions)
        train_f1_05 = metrics.f1_score(targets,(correct_predictions >= 0.5)*1)
        train_acc_05 = metrics.accuracy_score(targets,(correct_predictions >= 0.5)*1)
        train_balanced_acc_05 = metrics.balanced_accuracy_score(targets,(correct_predictions >= 0.5)*1)
    
    train_metrics = {
        'train_loss' : train_loss.avg,
        'train_auc' : train_auc,
        'train_f1_05' : train_f1_05,
        'train_acc_05' : train_acc_05,
        'train_balanced_acc_05' : train_balanced_acc_05,
        'epoch' : epoch
    }
    wandb.log(train_metrics)
    
    return train_metrics
    
    
def valid_epoch(model, val_data_loader, criterion, epoch):
    model.eval()
    
    valid_loss = AverageMeter()
    correct_predictions = []
    targets = []
    example_images = []
    
    
    with torch.no_grad():   
        idx = 1     
        for batch in tqdm(val_data_loader):
            # batch_image_names = batch['image_name']
            batch_images = batch['image'].to(device).float()
            batch_labels = batch['label'].to(device).float()
            
            out = model(batch_images)
            loss = criterion(out, batch_labels.view(-1, 1).type_as(out))
            
            valid_loss.update(loss.item(), val_data_loader.batch_size)
            batch_targets = (batch_labels.view(-1,1).cpu() >= 0.5) *1
            batch_preds = torch.sigmoid(out).cpu()
            
            targets.append(batch_targets)
            correct_predictions.append(batch_preds)
                
            best_batch_pred_idx = np.argmin(abs(batch_targets - batch_preds))  
            worst_batch_pred_idx = np.argmax(abs(batch_targets - batch_preds))
            example_images.append(wandb.Image(batch_images[best_batch_pred_idx],
                                      caption=f"Pred : {batch_preds[best_batch_pred_idx].item()} Label: {batch_targets[best_batch_pred_idx].item()}"))
            
            example_images.append(wandb.Image(batch_images[worst_batch_pred_idx],
                                      caption=f"Pred : {batch_preds[worst_batch_pred_idx].item()} Label: {batch_targets[worst_batch_pred_idx].item()}"))

            if(idx % 100 == 0):
                temp_t = np.vstack((targets)).ravel()
                temp_correct_preds = np.vstack((correct_predictions)).ravel()

                valid_auc = metrics.roc_auc_score(temp_t, temp_correct_preds)
                valid_f1_05 = metrics.f1_score(temp_t,(temp_correct_preds >= 0.5)*1)
                valid_acc_05 = metrics.accuracy_score(temp_t,(temp_correct_preds >= 0.5)*1)
                valid_balanced_acc_05 = metrics.balanced_accuracy_score(temp_t,(temp_correct_preds >= 0.5)*1)

                valid_metrics = {
                    'b_valid_loss' : valid_loss.avg,
                    'b_valid_auc' : valid_auc,
                    'b_valid_f1_05' : valid_f1_05,
                    'b_valid_acc_05' : valid_acc_05,
                    'b_valid_balanced_acc_05' : valid_balanced_acc_05,
                    'b_valid_batch' : idx
                }
                wandb.log(valid_metrics)
            idx += 1
    
    # Logging
    targets = np.vstack((targets)).ravel()
    correct_predictions = np.vstack((correct_predictions)).ravel()

    valid_auc = metrics.roc_auc_score(targets, correct_predictions)
    valid_f1_05 = metrics.f1_score(targets,(correct_predictions >= 0.5)*1)
    valid_acc_05 = metrics.accuracy_score(targets,(correct_predictions >= 0.5)*1)
    valid_balanced_acc_05 = metrics.balanced_accuracy_score(targets,(correct_predictions >= 0.5)*1)
    
    valid_metrics = {
        'valid_loss' : valid_loss.avg,
        'valid_auc' : valid_auc,
        'valid_f1_05' : valid_f1_05,
        'valid_acc_05' : valid_acc_05,
        'valid_balanced_acc_05' : valid_balanced_acc_05,
        'valid_examples' : example_images[-50:],
        'epoch' : epoch
    }
    wandb.log(valid_metrics)

    return valid_metrics
    
def create_train_transforms(size=224):
    return Compose([
        ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    ) 
    
def create_val_transforms(size=224):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

    
if __name__ == "__main__":
    run = 1
    model_name = 'xception'
    train(name='CelebDF_hardcore,'+model_name, val_fold=1, run=run, folds_csv='folds.csv')