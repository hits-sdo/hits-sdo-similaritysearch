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
from dataset import AIAHMItilesDataset
from lightly.data import LightlyDataset
import os

import sys
sys.path.append('/home/subhamoy/search/hits-sdo-similaritysearch/')
from search_simsiam.custom_collate import sunbirdCollate
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml
import glob
from tqdm import tqdm
import wandb
import random
from model import SimSiam

def main():
    
    seed = 23
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('high')
    
    with open('config.yml','r') as f:
        config_data = yaml.safe_load(f)
    
    instr = config_data['instr']
    contrastive = config_data['contrastive']
    gpu_number = config_data['gpu_number']
    num_workers = config_data['num_workers']                                                                  # How many process giving model to train -- similar to threading
    batch_size = config_data['batch_size']                                                                  # A subset of files that the model sees to update it's parameters                                                                        # Seed for random generator for reproducability
    epochs = config_data['epochs']                                                                    # How many times we go through our entire data set                                                               # The number of pixels in x or y
    num_ftrs = config_data['num_ftrs']
    stride = config_data['stride']                                                                      # Dimension of the embeddings
    fill_type = config_data['fill_type']
    iterative = False
    lr = 0.05 * batch_size / 256
    optim_type = config_data['optim_type']
    lr_schedule_coeff = config_data['']
    pretrained = config_data['pretrained']
    # Dimension of the output of the prediction and projection heads
    out_dim = config_data['out_dim']
    proj_hidden_dim = config_data['proj_hidden_dim']

    # The prediction head uses a bottleneck architecture
    pred_hidden_dim = config_data['pred_hidden_dim']
    offlimb_frac = config_data['offlimb_frac']
    
    name = f"Subh_SIMSIAM_ftrs_{num_ftrs}_pretrained_{pretrained}_"\
        "projdim_{proj_hidden_dim}_preddim_{pred_hidden_dim}_"\
            "odim_{out_dim}_contrastive_{contrastive}_"\
                "AIA_211_193_171_patch_stride_{stride}_batch_{batch_size}_"\
                    "optim_{optim_type}_lr_{lr}_"\
                        "schedule_coeff_{lr_schedule_coeff}_offlimb_frac_{offlimb_frac}"

    DATA_DIR = config_data['data_dir'][instr]
    MODEL_DIR = config_data['model_dir'][instr]
    
    dataset_train_simsiam = AIAHMItilesDataset(data_path=DATA_DIR, augmentation='double',
                                               data_stride=stride, offlimb_frac=offlimb_frac,
                                               step_multiplier=1,
                                               batch_size=batch_size,
                                               instr=instr)
    
    
    checkpoint_callback = ModelCheckpoint(dirpath=MODEL_DIR,
                                    filename=name,
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
                                "fill_type": fill_type,
                                "iterative filling": iterative,
                                "data_stride": stride,
                                "wavelengths": instr
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
                         callbacks=[checkpoint_callback, plotter, lr_monitor],
                         log_every_n_steps=5)

    trainer.fit(model=model, train_dataloaders=dataloader_train_simsiam)
    wandb.finish()

if __name__=='__main__':
    main()