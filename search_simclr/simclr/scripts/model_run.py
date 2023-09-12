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
from search_simclr.simclr.model.simCLR import SimCLR, SimCLRCallback
from search_simclr.simclr_utils.vis_utils import generate_embeddings, plot_knn_examples, plot_nearest_neighbors_3x3
from typing import Tuple
from argparse import ArgumentParser
import yaml
from datetime import datetime
from search_simclr.simclr.scripts.sdoconfig_dataclass import SDOConfig

def train(sweep = True):
    config = SDOConfig()
    parser = ArgumentParser()
    parser.add_argument("--model",type=str,help="The model to initialize.",default=config.model)
    parser.add_argument("--backbone",type=str,help="The backbone to use in model (resnet18,resnet34,resnet50,resnet101,resnet152,densenet121)",default=config.backbone)
    parser.add_argument("--lr",type=float,help="The learning rate for training model",default=config.lr)
    parser.add_argument("--epochs",type=int,help="Number of epochs to train for.",default=config.epochs)
    parser.add_argument("--split",type=bool,help="True if you want to overide the split files",default=True)
    parser.add_argument("--percent",type=float,help="Percentage of the total number of files that's reserved for training",default=config.percent_split)
    parser.add_argument("--numworkers",type=int,help="Number of processors running at the same time",default=config.num_workers)
    parser.add_argument('--tile_dir', type=str, default=config.tile_dir, help='Path to tile directory')
    parser.add_argument('--train_dir', type=str, default=config.train_dir, help='Path to train directory')
    parser.add_argument('--val_dir', type=str, default=config.val_dir, help='Path to validation directory') 
    parser.add_argument('--test_dir', type=str, default=config.test_dir, help='Path to test directory')
    parser.add_argument('--save_checkpoint_dir', type=str,default=config.save_checkpoint_dir, help="Path to checkpoint directory")
    parser.add_argument('--train_fpath', type=str, default=config.train_fpath, help='Path to train file list')
    parser.add_argument('--val_fpath', type=str, default=config.val_fpath, help='Path to validation file list')
    parser.add_argument('--test_fpath', type=str, default=config.test_fpath, help='Path to test file list')
    parser.add_argument('--percent_split', type=float, default=config.percent_split, help='Percentage split for train/val') 
    parser.add_argument('--num_img', type=int, default=config.num_img, help='Number of images')
    parser.add_argument('--save_vis_dir', type=str, default=config.save_vis_dir, help='Path to save visualizations')
    parser.add_argument('--save_model_dir', type=str, default=config.save_model_dir, help='Path to save models')
    parser.add_argument('--tot_fpath_wfname', type=str, default=config.tot_fpath_wfname, help='Path to total file list')
    parser.add_argument('--blur', type=Tuple[int, int], default=config.blur, help='Blur range')
    parser.add_argument('--brighten', type=float, default=config.brighten, help='Brightness level')
    parser.add_argument('--translate', type=Tuple[int,int], default=config.translate, help='Translate range')
    parser.add_argument('--zoom', type=float, default=config.zoom, help='Zoom level')
    parser.add_argument('--rotate', type=float, default=config.rotate, help='Rotation angle') 
    parser.add_argument('--noise_mean', type=float, default=config.noise_mean, help='Noise mean')
    parser.add_argument('--noise_std', type=float, default=config.noise_std, help='Noise standard deviation')
    parser.add_argument('--cutout_holes', type=int, default=config.cutout_holes, help='Number of cutout holes')
    parser.add_argument('--cutout_size', type=float, default=config.cutout_size, help='Cutout size')
    parser.add_argument('--num_workers', type=int, default=config.num_workers, help='Number of workers')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size')
    parser.add_argument('--seed', type=int, default=config.seed, help='The seed to fill the model with')
    parser.add_argument('--input_size', type=int, default=config.input_size, help='Input resolution') 
    parser.add_argument('--num_ftrs', type=int, default=config.num_ftrs, help='Number of features (for pca and tsne). Set the number of features (how many axes of an object) to compress data to')
    parser.add_argument('--accelerator', type=str, default=config.accelerator, help='Device to run the model from (GPU or CPU)')
    parser.add_argument('--devices', type=bool, default=config.devices, help='Use GPUs if available')
    parser.add_argument('--train_stage', type=str, default=config.train_stage, help='Stage the model for training')
    parser.add_argument('--val_stage', type=str, default=config.val_stage, help='Stage the model for validation')
    parser.add_argument('--enable_checkpoint', type=bool, default=config.enable_checkpoint, help='Enable checkpointing')
    parser.add_argument('--log_every_n_steps', type=int, default=config.log_every_n_steps, help='Log every n steps')
    # parser.add_argument('--sweep', type=bool, default=config.val_stage, help='Stage the model for validation')

    args, _ = parser.parse_known_args()


    if not os.path.exists(args.save_model_dir):
        raise Exception("visual directory not defined.")
    if not os.path.exists(args.save_vis_dir):
        raise Exception("visual directory not defined.")
    if not os.path.exists(args.save_checkpoint_dir):
        raise Exception("checkpoint directory not defined.")
        
    # Todo: Add "model_backbone" argument
    # Add: cpus, val split, gpus
 
    # Seed pytorch lightning
    pl.seed_everything(args.seed)
    
    wandb.init(project="search_simclr",
        dir=args.save_vis_dir,
        # config="sweeps.yml"
        # config=
        # {
        #     "architecture": "SimCLR",
        #     "dataset": "miniset2.0",
        #     "epochs": args.epochs,
        # }
    )
    
    # Setup wandb config
    
    
    if (sweep):
        wandb_config = wandb.config
    #if (wandb_config is not None):
        args.lr = wandb_config.learning_rate
        args.batch_size = wandb_config.batch_size
        args.epochs = wandb_config.epochs
        args.backbone = wandb_config.backbone
        print("Using wandb config: "+str(wandb_config) + "\n")
    else:
        print("Not using wandb config")


    
    # save_vis_dir: str = os.path.join(root, "ssearch_simclr", "visualizations", "simclr_knn")
    # save_model_dir: str = os.path.join(root, "search_simclr", "model_weights")
    # train_flist = get_file_list(args.train_fpath)
    # val_flist = get_file_list(args.val_fpath)
    # Split the data into train and val
  


    # print(f'train_flist: {train_flist}')
    # ❗ problem: datamodule now takes in lists 
    sdo_datamodule = SimCLRDataModule(
                 blur = args.blur, 
                 brighten = args.brighten, 
                 translate = args.translate, 
                 zoom = args.zoom, 
                 rotate = args.rotate, 
                 noise_mean = args.noise_mean, 
                 noise_std = args.noise_std, 
                 cutout_holes = args.cutout_holes, 
                 cutout_size = args.cutout_size,
                 batch_size = args.batch_size,
                 num_images = args.num_img,
                 percent = args.percent,
                 
                 tile_dir = args.tile_dir,
                 train_dir = args.train_dir,
                 val_dir = args.val_dir,
                 test_dir = args.test_dir,
                 
                 train_fpath = args.train_fpath,
                 val_fpath = args.val_fpath,
                 train_flist = None, 
                 val_flist = None, #todo
                 test_flist = None,
                 tot_fpath_wfname = args.tot_fpath_wfname,
                 split = args.split,
                 num_workers = args.num_workers)
    sdo_datamodule.prepare_data()
    sdo_datamodule.setup(stage=args.train_stage)


    for batch_idx, (img1, img2, fname, _) in enumerate(sdo_datamodule.train_dataloader()):
        print (batch_idx, img1.shape, img2.shape, fname)
        break
    
    #success = wandb.login()
    #print("Wandb: "+success)
    
    #accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    # offset = random.random() / 5

    # simulating a training run
    # for epoch in range(2, args.epochs):
    #     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    #     loss = 2 ** -epoch + random.random() / epoch + offset
    #     print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    #     wandb.log({"accuracy": acc, "loss": loss})

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Training the Model
    model = SimCLR(args.lr, args.backbone)
    # model = model.to(torch.float64)
    callback = SimCLRCallback(os.path.join(args.save_checkpoint_dir, f'{now_str}_{args.backbone}_embeddings.pt'))
    trainer = pl.Trainer(max_epochs=args.epochs, devices=args.devices, accelerator=args.accelerator, log_every_n_steps=args.log_every_n_steps, default_root_dir=args.save_checkpoint_dir, enable_checkpointing=args.enable_checkpoint, callbacks=[callback])
    trainer.fit(model, sdo_datamodule.train_dataloader())
    
    # Save the Model
    trained_backbone = model.backbone
    state_dict = {"backbone_state_dict": trained_backbone.state_dict()}
    
    torch.save(state_dict, os.path.join(args.save_model_dir, f'{now_str}_model.pth'))
    
    # Running validation
    sdo_datamodule.setup(stage=args.val_stage)
    model.eval()
    # embeddings, filenames = generate_embeddings(model, sdo_datamodule.val_dataloader()) 
    # Todo: Once when have the test dataset, replace val_dataloader with test_dataloader
    # plot_knn_examples(embeddings, filenames, path_to_data=args.tile_dir, n_neighbors=3, num_examples=6, vis_output_dir=args.save_vis_dir)
    
    # Todo: TEST CODE FROM SIMSIAM
    # example_images = [filenames[10**n] for n in range(5)]
    # # show example images for each cluster
    # for i, example_image in enumerate(example_images):
    #     plot_nearest_neighbors_3x3(example_image, i)
    
    wandb.finish()

def main():
    train(False)
    
    # trainer.fit(model, dataloader_train_simclr)
if __name__ == "__main__":
    main()
    