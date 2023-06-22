import sys,os
sys.path.append(os.getcwd())

import numpy as np
import torchvision
from torch import nn
import wandb
from sklearn import random_projection
from sklearn.preprocessing import MinMaxScaler, normalize
from src.data import TilesDataModule
from src.model import BYOL, SimSiam
from search_utils.analysis_utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import torch
import yaml

torchvision.disable_beta_transforms_warning()

def main():    
    # read in config file
    with open('experiment_config.yml') as config_file:
        config = yaml.safe_load(config_file.read())
    

    run = wandb.init(config=config,project=config['meta']['project'],
                        name=config['meta']['name'],
                        group=config['meta']['group'],
                        tags=config['meta']['tags'])
    config = wandb.config

    # set seeds
    pl.seed_everything(42,workers=True)
    torch.set_float32_matmul_precision('high')

    # define data module
    data = TilesDataModule(data_path=config.data['data_path'],
                           batch=config.training['batch_size'],
                           augmentation=config.data['augmentation'])

    # initialize model
    model = BYOL(lr=config.training['lr'],
                 wd=config.training['wd'],
                 input_channels=config.model['channels'],
                 projection_size=config.model['projection_size'],
                 prediction_size=config.model['prediction_size'],
                 cosine_scheduler_start=config.training['momentum_start'],
                 cosine_scheduler_end=config.training['momentum_end'],
                 loss=config.training['loss'])

    # initialize wandb logger
    wandb_logger = WandbLogger(log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor='loss',
                                          mode='min',
                                          save_top_k=2,
                                          save_last=True,
                                          save_weights_only=True,
                                          verbose=False)

    # train model
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=config.training['epochs'],
                         log_every_n_steps=50,
                        #  limit_train_batches=5,
                        #  limit_test_batches=1,
                         logger=wandb_logger,
                         deterministic=True)
    trainer.fit(model=model,datamodule=data)

    # save predictions
    preds_train = trainer.predict(model=model,dataloaders=data.train_dataloader())
    files_train, embeddings_train,embeddings_proj_train = save_predictions(preds_train,wandb.run.dir,'train')

    # normalize and project embeddings into 2D for plotting
    projection = random_projection.GaussianRandomProjection(n_components=2)
    embeddings_2d_train = projection.fit_transform(embeddings_train)    
    
    scaler = MinMaxScaler()
    embeddings_2d_train = scaler.fit_transform(embeddings_2d_train)
    fig = get_scatter_plot_with_thumbnails(embeddings_2d_train,files_train)
    wandb.log({"Backbone_embeddings_2D": wandb.Image(fig)})

    scaler2 = MinMaxScaler()
    embeddings_proj_train = scaler2.fit_transform(embeddings_proj_train)
    fig2 = get_scatter_plot_with_thumbnails(embeddings_proj_train,files_train)
    wandb.log({"Projection_embeddings_2D": wandb.Image(fig2)})

    wandb.finish()

if __name__ == "__main__":
    main()
