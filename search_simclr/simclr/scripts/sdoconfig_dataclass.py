import os
import torch
from typing import Tuple
import pyprojroot
root = pyprojroot.here()
from dataclasses import dataclass


@dataclass
class SDOConfig:
    """ Configuration options for HITS-SDO Dataset"""
    tile_dir = os.path.join(root , 'data')
    train_dir: str = os.path.join(tile_dir, 'train_val_simclr')
    val_dir: str = os.path.join(tile_dir, 'train_val_simclr')
    test_dir: str = None
    train_fpath: str = os.path.join(train_dir,'train_file_list.txt')
    val_fpath: str = os.path.join(val_dir, 'val_file_list.txt')
    test_fpath: str = None
    percent_split: float = 0.8
    num_img: int = 1500
    model: str = "simclr"
    backbone: str = "resnet18"
    
    save_vis_dir: str = os.path.join(root, "search_simclr", "visualizations", "simclr_knn")
    save_model_dir: str = os.path.join(root, "search_simclr", "model_weights")
    save_checkpoint_dir: str = os.path.join(root, "search_simclr", "checkpoints")
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

    lr: float = 0.005
    num_workers: int = 12
    batch_size: int = 4
    seed: int = 1
    epochs: int = 3
    input_size: int = 128 # input resolution
    num_ftrs: int = 32
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"
    devices: bool = 1
    train_stage: str = "train"
    val_stage: str = "validate"    
    enable_checkpoint: bool = True
    log_every_n_steps: int = 10