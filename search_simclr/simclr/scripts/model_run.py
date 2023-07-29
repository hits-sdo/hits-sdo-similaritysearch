import os, random, shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import os.path
import pyprojroot
root = pyprojroot.here()
utils_dir = root / 'search_utils'
import sys
import wandb
import pytorch_lightning as pl
sys.path.append(str(root))
from dataclasses import dataclass
from search_utils import image_utils
from search_simclr.simclr.dataloader import dataset_aug
from search_simclr.simclr.dataloader.dataset import SdoDataset, partition_tile_dir_train_val
from search_utils.file_utils import get_file_list
from search_simclr.simclr.dataloader.datamodule import SimCLRDataModule
from search_simclr.simclr.model.simCLR import SimCLR


@dataclass
class SDOConfig:
    """ Configuration options for HITS-SDO Dataset"""

    train_dir: str = root/"data/miniset/AIA171/monochrome" # ❗want train_list?
    val_dir: str = None # ❗want val_list instead?
    test_dir: str = None
    save_vis_dir: str = os.path.join(root, "search_simclr", "visualizations", "simclr_knn")
    save_model_dir: str = os.path.join(root, "search_simclr", "model_weights")
    #TODO: train_flist: str = 
    num_workers: int = 8
    batch_size: int = 256
    seed: int = 1
    max_epochs: int = 20
    input_size: int = 128 # input resolution
    num_ftrs: int = 32
    accelerator: str = "cuda" if torch.cuda.is_available() else "cpu"
    devices: bool = 1
    stage: str = "train"

def main():
    config = SDOConfig()
    pl.seed_everything(config.seed)
    # ❗ problem: datamodule now takes in lists 
    sdo_datamodule = SimCLRDataModule(batch_size=config.batch_size, train_dir=config.train_flist=)
    sdo_datamodule.setup(stage=config.stage)

    for batch_idx, (img1, img2, fname, _) in enumerate(sdo_datamodule.train_dataloader()):
        print (batch_idx, img1.shape, img2.shape, fname)
        break

    #accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    wandb.init(project="SimCLR",
                dir=config.save_vis_dir,
                config=
                {
                    "architecture": "SimCLR",
                    "dataset": "miniset2.0",
                    "epochs": config.max_epochs,
                }
    )

    model = SimCLR()
    trainer = pl.Trainer(max_epochs=config.max_epochs, devices=config.devices, accelerator=config.accelerator)
    trainer.fit(model, sdo_datamodule.train_dataloader())
    
    # trainer.fit(model, dataloader_train_simclr)
if __name__ == "__main__":
    main()
    