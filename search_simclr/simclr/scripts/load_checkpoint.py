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

def main():
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file name')
    args = parser.parse_args()

    # Validate path
    if (args.checkpoint is None):
        raise Exception("Checkpoint file name not defined.")
    path = os.path.join(root, 'search_simclr', 'checkpoints', args.checkpoint)
    if not os.path.exists(path):
        raise Exception("File not found: "+path)

    # Load the model
    model = SimCLR().load_from_checkpoint(path)
    print("Model Learning Rate: "+str(model.lr))

    # Test 1:
    # checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    # print(checkpoint["hyper_parameters"])
    # {"learning_rate": the_value, "another_parameter": the_other_value}

    # Test 2:
    print("Loading checkpoint keys:\n")
    checkpoint = torch.load(path)
    print(checkpoint["hyper_parameters"])
    # print(checkpoint["state_dict"])
    print(checkpoint.keys())

    # disable randomness, dropout, etc...
    model.eval()

    # load images from folder
    folder = os.path.join(root, 'search_simclr', 'simclr', 'scripts', 'test_images')
    if not os.path.exists(folder):
        raise Exception("test images directory not defined.")

    images = []
    for filename in os.listdir(folder):
        img = image_utils.read_image(os.path.join(folder, filename), 'jpg')
        if img is not None:
            images.append(toTensor(img))
        else:
            print("Error reading image: "+filename)

    # predict with the model
    # img = torch.from_numpy(img)
    representations = model(img)

    print("Representations shape: "+str(representations.shape))
    print("Representations: "+str(representations))

    # Generate embeddings for validation

def toTensor(image):
    # If the image is grayscale, expand its dimensions to have a third axis
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)

if __name__ == '__main__':
    main()