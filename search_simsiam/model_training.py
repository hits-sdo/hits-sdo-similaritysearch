import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
torch.set_float32_matmul_precision("high")
from plotter import SimSiamPlotter
import matplotlib.pyplot as plt
from dataset import HMItilesDataset
from lightly.data import LightlyDataset
import os

import sys
sys.path.append('/home/subhamoy/search/hits-sdo-similaritysearch/')
from search_simsiam.custom_collate import sunbirdCollate
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import glob
from tqdm import tqdm
import wandb
import random
from model import SimSiam

def main():
    
    seed = 23
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('high')
    
    instr = 'euv'
    contrastive = False
    gpu_number = 1
    num_workers = 4                                                                  # How many process giving model to train -- similar to threading
    batch_size = 64                                                                  # A subset of files that the model sees to update it's parameters                                                                        # Seed for random generator for reproducability
    epochs = 10                                                                       # How many times we go through our entire data set                                                               # The number of pixels in x or y
    num_ftrs = 512
    stride = 1                                                                      # Dimension of the embeddings
    fill_type = "Nearest"
    iterative = False
    if fill_type == 'Nearest': iterative = False
    lr = 0.05 * batch_size / 256
    optim_type = 'sgd'
    lr_schedule_coeff = 1.0
    pretrained = False
    # Dimension of the output of the prediction and projection heads
    out_dim = proj_hidden_dim = 512 #64

    # The prediction head uses a bottleneck architecture
    pred_hidden_dim = 128 #32
    offlimb_frac = 1
    # lr_schedule = True
    
    name = f"Subh_SIMSIAM_ftrs_{num_ftrs}_pretrained_{pretrained}_projdim_{proj_hidden_dim}_preddim_{pred_hidden_dim}_odim_{out_dim}_contrastive_{contrastive}_AIA_211_193_171_patch_stride_{stride}_batch_{batch_size}_optim_{optim_type}_lr_{lr}_schedule_coeff_{lr_schedule_coeff}_offlimb_frac_{offlimb_frac}"

    if instr.lower()=='mag':
        DATA_DIR = '/d0/euv/aia/preprocessed/HMI/HMI_256x256/'
        MODEL_DIR = '/d0/subhamoy/models/search/magnetograms/'
    else:
        DATA_DIR = '/d0/euv/aia/preprocessed/AIA_211_193_171/AIA_211_193_171_256x256/'
        MODEL_DIR = '/d0/subhamoy/models/search/AIA_211_193_171/'
    
    # dataset_train_simsiam = LightlyDataset(input_dir=DATA_DIR)
    
    # if stride>1:
    #     indices = range(0, len(dataset_train_simsiam), stride)
    #     dataset_train_simsiam = torch.utils.data.Subset(dataset_train_simsiam, indices)
    dataset_train_simsiam = HMItilesDataset(data_path=DATA_DIR, augmentation='double',
                                               data_stride=stride, offlimb_frac=offlimb_frac,
                                               step_multiplier=1,
                                               batch_size=batch_size,
                                               instr=instr)
        
    # sun_bird_collate_fn = sunbirdCollate(path_to_data=MODEL_DIR, 
    #                                     fill_type='Nearest', multi_wl=True,
    #                                     iterative=False)
    
    
    checkpoint_callback = ModelCheckpoint(dirpath=MODEL_DIR,
                                    filename=name, #'{epoch}-{name}',
                                    save_top_k=1,
                                    verbose=True,
                                    monitor='loss',
                                    mode='min')
    
    plotter = SimSiamPlotter(stride=stride, instr=instr)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    dataloader_train_simsiam = torch.utils.data.DataLoader(
        dataset_train_simsiam,
        batch_size=batch_size,
        shuffle=True,                                                                   
        # collate_fn=sun_bird_collate_fn,                                                 
        drop_last=True,                                                                 
        num_workers=num_workers
    )
        
    wandb_logger = WandbLogger(entity="sc8473",
                            # Set the project where this run will be logged
                            project="search",
                            name = name,
                            # Track hyperparameters and run metadata
                            config={
                                "learning_rate": lr,
                                "epochs": epochs,
                                "batch_size": batch_size,
                                "collate_fn": "sunbird_collate",
                                "fill_type": fill_type,
                                "iterative filling": iterative,
                                "data_stride": stride,
                                "wavelengths": "magnetogram"
                            })
    
    model = SimSiam(num_ftrs=num_ftrs, proj_hidden_dim=proj_hidden_dim,
                    pred_hidden_dim=pred_hidden_dim, out_dim=out_dim,
                    lr=lr, lr_schedule_coeff=lr_schedule_coeff,
                    optim_type=optim_type, pretrained=pretrained,
                    contrastive=contrastive)
    
    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator='gpu',
                         devices=[gpu_number],
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, plotter, lr_monitor], #
                         log_every_n_steps=5)

    trainer.fit(model=model, train_dataloaders=dataloader_train_simsiam)
    wandb.finish()
if __name__=='__main__':
    main()