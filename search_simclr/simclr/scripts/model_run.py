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
from search_simclr.simclr_utils.vis_utils import generate_embeddings, plot_knn_examples
from typing import Tuple


@dataclass
class SDOConfig:
    """ Configuration options for HITS-SDO Dataset"""
    tile_dir = os.path.join(root , 'data')
    train_dir: str = os.path.join(tile_dir, 'train_val_simclr')
    val_dir: str = os.path.join(tile_dir, 'train_val_simclr')
    test_dir: str = None
    train_flist: str = os.path.join(train_dir,'train_file_list.txt')
    val_flist: str = os.path.join(val_dir, 'val_file_list.txt')
    test_flist: str = None
    
    save_vis_dir: str = os.path.join(root, "search_simclr", "visualizations", "simclr_knn")
    save_model_dir: str = os.path.join(root, "search_simclr", "model_weights")
    #TODO: train_flist: str = 
    tot_fpath_wfname = os.path.join(train_dir, 'tot_full_path_files.txt')
    blur: Tuple[int, int] = (5,5)
    brighten: float = 1.0
    translate: Tuple[int, int] = (1,3)
    zoom: float = 1.5
    rotate: float = 360.0
    noise_mean: float = 0.0 
    noise_std: float = 0.05
    cutout_holes: int = 1 
    cutout_size: float = 0.3

    num_workers: int = 20
    batch_size: int = 5
    seed: int = 1
    epochs: int = 20
    input_size: int = 128 # input resolution
    num_ftrs: int = 32
    accelerator: str = "cuda" if torch.cuda.is_available() else "cpu"
    devices: bool = 1
    train_stage: str = "train"
    val_stage: str = "validate"

def main():
    config = SDOConfig()
    pl.seed_everything(config.seed)
    
    print(f"train_flist[0] = {config.train_flist[0]}")
    
    # ❗ problem: datamodule now takes in lists 
    sdo_datamodule = SimCLRDataModule(
                 blur = config.blur, 
                 brighten = config.brighten, 
                 translate = config.translate, 
                 zoom = config.zoom, 
                 rotate = config.rotate, 
                 noise_mean = config.noise_mean, 
                 noise_std = config.noise_std, 
                 cutout_holes = config.cutout_holes, 
                 cutout_size = config.cutout_size,
                 batch_size = config.batch_size,
                 
                 tile_dir = config.tile_dir,
                 train_dir = config.train_dir,
                 val_dir = config.val_dir,
                 test_dir = config.test_dir,
                 
                 train_flist = config.train_flist, 
                 val_flist = config.val_flist, 
                 test_flist = config.test_flist,
                 tot_fpath_wfname = config.tot_fpath_wfname,
                 num_workers = config.num_workers)
    sdo_datamodule.setup(stage=config.train_stage)

    for batch_idx, (img1, img2, fname, _) in enumerate(sdo_datamodule.train_dataloader()):
        print (batch_idx, img1.shape, img2.shape, fname)
        break
    
    #success = wandb.login()
    #print("Wandb: "+success)
    
    #accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    wandb.init(project="SimCLR",
                dir=config.save_vis_dir,
                config=
                {
                    "architecture": "SimCLR",
                    "dataset": "miniset2.0",
                    "epochs": config.epochs,
                }
    )
    
    offset = random.random() / 5

    # simulating a training run
    # for epoch in range(2, config.epochs):
    #     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    #     loss = 2 ** -epoch + random.random() / epoch + offset
    #     print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    #     wandb.log({"accuracy": acc, "loss": loss})

    

    model = SimCLR()
    trainer = pl.Trainer(max_epochs=config.epochs, devices=config.devices, accelerator=config.accelerator)
    trainer.fit(model, sdo_datamodule.train_dataloader())
    
    # Todo: Make a better way to change the variable
    sdo_datamodule.setup(stage=config.val_stage)
    model.eval()
    # embeddings, filenames = generate_embeddings(model, sdo_datamodule.val_dataloader())
    # plot_knn_examples(embeddings, filenames)
    wandb.finish()
    # trainer.fit(model, dataloader_train_simclr)
if __name__ == "__main__":
    main()
    