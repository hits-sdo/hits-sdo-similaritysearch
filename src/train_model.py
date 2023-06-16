import sys,os
sys.path.append(os.getcwd())

import numpy as np

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
                         log_every_n_steps=1,
                         limit_train_batches=10,
                         limit_test_batches=4,
                         logger=wandb_logger,
                         deterministic=True)
    trainer.fit(model=model,datamodule=data)

    # save predictions locally
    preds = trainer.predict(model=model,dataloaders=data.test_dataloader())
    file = []
    embeddings = []
    for predbatch in preds:
        file.extend(predbatch[0])
        embeddings.extend(np.array(predbatch[1]))
    df = pd.DataFrame({'filename':file,'embedding':embeddings})
    df.to_csv(wandb.run.dir+os.sep+'embeddings.csv',index=False)

    wandb.finish()

if __name__ == "__main__":
    main()
