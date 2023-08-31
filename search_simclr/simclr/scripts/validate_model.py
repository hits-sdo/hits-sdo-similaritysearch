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
from search_simclr.simclr_utils.vis_utils import generate_embeddings, plot_knn_examples, plot_nearest_neighbors_3x3
from typing import Tuple
from argparse import ArgumentParser
import yaml
from datetime import datetime

# Load the model from path
lr = 0.0003
model = SimCLR(lr)
path = os.path.join(root, 'search_simclr', 'model_weights', '2023-08-28_10-05-25PeachySweepmodel.pth')
model.load_state_dict(torch.load(path))

# Validate the model

# Generate embeddings for validation
