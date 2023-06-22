import sys,os
sys.path.append(os.getcwd())

import numpy as np
import torchvision
from torch import nn
import wandb
from src.data import TilesDataModule
from src.model import BYOL, SimSiam
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
                 cosine_scheduler_end=config.training['momentum_end'])

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
    preds = trainer.predict(model=model,dataloaders=data.test_dataloader())
    file = []
    embeddings = []
    embeddings_proj = []
    for predbatch in preds:
        file.extend(predbatch[0])
        embeddings.extend(np.array(predbatch[1]))
        embeddings_proj.extend(np.array(predbatch[2]))

    np.save(wandb.run.dir+os.sep+'embeddings.npy',np.array(embeddings))
    np.save(wandb.run.dir+os.sep+'embeddings_proj.npy',np.array(embeddings_proj))
    df = pd.DataFrame({'filename':file})
    df.to_csv(wandb.run.dir+os.sep+'filenames.csv',index=False)

    wandb.finish()

if __name__ == "__main__":
    main()
