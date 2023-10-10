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
from search_simclr.simclr_utils.vis_utils import (
    generate_embeddings,
    plot_scatter,
    plot_NxN_examples
)
# from typing import Tuple
from argparse import ArgumentParser
import yaml
from datetime import datetime
from search_simclr.simclr.scripts.sdoconfig_dataclass import SDOConfig
from search_simclr.simclr.dataloader.datamodule import SimCLRDataModule
from search_simclr.simclr_utils.embeddings import (
    performTSNE,
    performPCA,
    performUMAP,
    create_embeddings_component_table
)

def main():
    config = SDOConfig()
    #sdo_datamodule = SimCLRDataModule() # <-- stuff here

        
    # Load the model from path
    model = SimCLR(model_str=config.backbone)
    #model = model.to(torch.float64)
    path = os.path.join(root, 'search_simclr', 'model_weights', '2023-09-25_10-27-17_model_resnet152_easy_sweep.pth')
    print(f'path: {path}')
    temp = torch.load(path)
    #print(f'temp.shape: {temp.shape}')
    temp = temp["backbone_state_dict"] # FIX ME <- "backbone_state_dict"
    model.backbone.load_state_dict(temp)


    # Validate the model
    model.eval()
    # Fetch the validation set from the dataloader
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
                    num_images = config.num_img,
                    percent = config.percent_split,
                    
                    tile_dir = config.tile_dir,
                    train_dir = config.train_dir,
                    val_dir = config.val_dir,
                    test_dir = config.test_dir,
                    
                    train_fpath = config.train_fpath,
                    val_fpath = config.val_fpath,
                    train_flist = None, 
                    val_flist = None,
                    test_flist = None,
                    tot_fpath_wfname = config.tot_fpath_wfname,
                    split = True,
                    num_workers = config.num_workers)


    sdo_datamodule.prepare_data()
    sdo_datamodule.setup(stage=config.val_stage)

    # Generate embeddings for validation from the deap learning base encoder featured in simclr (resnet18 ... etc)
    embeddings, filenames = generate_embeddings(model, sdo_datamodule.val_dataloader())
    # take embeddings output from base encoder, and apply dimentionality reduction, to plot the embedding space in 2d
    # perform_tsne, perform_gausiona, perform_pca, perform_umap ->write the object as a table

    print("Lengths - Embeddings: ", len(embeddings), " Filenames: ", len(filenames))

    num_components = 2

    # embeddings after dimensionality reduction
    # embeddings_tsne = performTSNE(embeddings, num_components)
    # embeddings_pca = performPCA(embeddings, num_components)
    # embeddings_umap = performUMAP(embeddings, num_components)
    
    # # Plot 2D Scatter Plots
    # plot(config.save_vis_dir, config.tile_dir, filenames, "embeddings_tsne", embeddings_tsne, num_components)
    # plot(config.save_vis_dir, config.tile_dir, filenames, "embeddings_pca", embeddings_pca, num_components)
    # plot(config.save_vis_dir, config.tile_dir, filenames, "embeddings_umap", embeddings_umap, num_components)
    
    # num_components = 3
    # embeddings_tsne = performTSNE(embeddings, num_components)
    # embeddings_pca = performPCA(embeddings, num_components)
    # embeddings_umap = performUMAP(embeddings, num_components)
    
    # # Plot 3D Scatter Plots
    # plot(config.save_vis_dir, config.tile_dir, filenames, "embeddings_tsne", embeddings_tsne, num_components)
    # plot(config.save_vis_dir, config.tile_dir, filenames, "embeddings_pca", embeddings_pca, num_components)
    # plot(config.save_vis_dir, config.tile_dir, filenames, "embeddings_umap", embeddings_umap, num_components)

    # Visualize Nearest Neighbors
    data_path = os.path.join(root, "data")
    # plot_knn_examples(embeddings, filenames, data_path, vis_output_dir=config.save_vis_dir)
    
    plot_NxN_examples(embeddings, filenames, data_path, vis_output_dir=config.save_vis_dir)
    plot_NxN_examples(embeddings, filenames, data_path, vis_output_dir=config.save_vis_dir, grid_size=5, num_examples=5, distType='cosine')
    plot_NxN_examples(embeddings, filenames, data_path, vis_output_dir=config.save_vis_dir, grid_size=7, num_examples=5, distType='euclidean')

def plot(vis_dir:str,
         tile_dir:str,
         filenames:List,
         title:str,
         dim_reduct_embed:np.ndarray,
         num_components:int) -> None:
    dim_reduct_df = create_embeddings_component_table(num_components, dim_reduct_embed, filenames, len(filenames))
    plot_scatter(dim_reduct_df, vis_dir, tile_dir, title)
    

if __name__ == "__main__":
    main()