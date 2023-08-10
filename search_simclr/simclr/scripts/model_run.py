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
from search_utils.file_utils import get_file_list, split_val_files
from search_simclr.simclr.dataloader.datamodule import SimCLRDataModule
from search_simclr.simclr.model.simCLR import SimCLR
from search_simclr.simclr_utils.vis_utils import generate_embeddings, plot_knn_examples
from typing import Tuple
from argparse import ArgumentParser


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
    num_img: int = None
    
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

    num_workers: int = 12
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
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="The model to initialize.",
        default="simclr"
        # Todo: add branch based on model
    )
    parser.add_argument(
        "--backbone",
        type=str,
        help="The backbone to use in model",
        default="resnet18"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        help="batch size to use for training model",
        default=50
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="The learning rate for training model",
        default=0.001
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train for.",
        default=20
    )
    parser.add_argument(
        "--split",
        type=bool,
        help="True if you want to ovveride the split files",
        default=False
    )
    parser.add_argument(
        "--percent",
        type=float,
        help="Percentage of the total number of files that's reserved for training",
        default=0.8
    )
    parser.add_argument(
        "--numworkers",
        type=int,
        help="Number of processors running at the same time",
        default=12
    )
    parser.add_argument('--tile_dir', type=str, default=os.path.join(root , 'data'), help='Path to tile directory')
    parser.add_argument('--train_dir', type=str, default=os.path.join(root, 'data', 'train_val_simclr'), help='Path to train directory')
    parser.add_argument('--val_dir', type=str, default=os.path.join(root, 'data', 'train_val_simclr'), help='Path to validation directory') 
    parser.add_argument('--test_dir', type=str, default=None, help='Path to test directory')
    parser.add_argument('--train_fpath', type=str, default=os.path.join(root, 'data', 'train_val_simclr','train_file_list.txt'), help='Path to train file list')
    parser.add_argument('--val_fpath', type=str, default=os.path.join(root, 'data', 'train_val_simclr', 'val_file_list.txt'), help='Path to validation file list')
    parser.add_argument('--test_fpath', type=str, default=None, help='Path to test file list')
    parser.add_argument('--percent_split', type=float, default=0.8, help='Percentage split for train/val') 
    parser.add_argument('--num_img', type=int, default=None, help='Number of images')
    parser.add_argument('--save_vis_dir', type=str, default=os.path.join(root, "search_simclr", "visualizations", "simclr_knn"), help='Path to save visualizations')
    parser.add_argument('--save_model_dir', type=str, default=os.path.join(root, "search_simclr", "model_weights"), help='Path to save models')
    parser.add_argument('--tot_fpath_wfname', type=str, default=os.path.join(root, 'data', 'train_val_simclr', 'tot_full_path_files.txt'), help='Path to total file list')
    parser.add_argument('--blur', type=Tuple[int, int], default=(5,5), help='Blur range')
    parser.add_argument('--brighten', type=float, default=1.0, help='Brightness level')
    parser.add_argument('--translate', type=Tuple[int,int], default=(1,3), help='Translate range')
    parser.add_argument('--zoom', type=float, default=1.5, help='Zoom level')
    parser.add_argument('--rotate', type=float, default=360.0, help='Rotation angle') 
    parser.add_argument('--noise_mean', type=float, default=0.0, help='Noise mean')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Noise standard deviation')
    parser.add_argument('--cutout_holes', type=int, default=1, help='Number of cutout holes')
    parser.add_argument('--cutout_size', type=float, default=0.3, help='Cutout size')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--seed', type=int, default=1, help='The seed to fill the model with')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--input_size', type=int, default=128, help='Input resolution') 
    parser.add_argument('--num_ftrs', type=int, default=32, help='Number of features (for pca and tsne). Set the number of features (how many axes of an object) to compress data to')
    parser.add_argument('--accelerator', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model from (GPU or CPU)')
    parser.add_argument('--devices', type=bool, default=1, help='Use GPUs if available')
    parser.add_argument('--train_stage', type=str, default='train', help='Stage the model for training')
    parser.add_argument('--val_stage', type=str, default='validate', help='Stage the model for validation')
    
    args, _ = parser.parse_known_args()
    print(args)
    # Todo: Add "model_backbone" argument
    # Add: cpus, val split, gpus
    
    
    
    config = SDOConfig()
    pl.seed_everything(config.seed)
    
    # Split the data into train and val
    if args.overrideSplitFiles or not (os.path.exists(config.train_fpath) and os.path.exists(config.val_fpath)):
        split_val_files(config.tot_fpath_wfname, config.strain_fpath, config.val_fpath, config.num_img, config.percent_split)
    
    # print(f"train_flist[0] = {config.train_flist[0]}")

    train_flist = get_file_list(config.train_fpath)
    val_flist = get_file_list(config.val_fpath)
    #test_flist = get_file_list(config.test_fpath)

    # print(f'train_flist: {train_flist}')
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
                 
                 train_flist = train_flist, 
                 val_flist = val_flist, 
                 test_flist = None,
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
    model = model.to(torch.float64)
    trainer = pl.Trainer(max_epochs=config.epochs, devices=config.devices, accelerator=config.accelerator)
    trainer.fit(model, sdo_datamodule.train_dataloader())
    
    # Todo: Make a better way to change the variable
    sdo_datamodule.setup(stage=config.val_stage)
    model.eval()
    # embeddings, filenames = generate_embeddings(model, sdo_datamodule.val_dataloader())
    # plot_knn_examples(embeddings, filenames)
    wandb.finish()

    trained_backbone = model.backbone
    state_dict = {"resnet18_parameters": trained_backbone.state_dict()}
    torch.save(state_dict, os.path.join(config.save_model_dir, "model.pth"))
    # trainer.fit(model, dataloader_train_simclr)
if __name__ == "__main__":
    main()
    