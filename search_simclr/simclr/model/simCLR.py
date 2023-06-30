import pyprojroot
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils

@dataclass
class SDOConfig:
    """ Configuration options for HITS-SDO Dataset"""

    train_dir: str = root/"data/miniset/AIA171/monochrome",
    val_dir: str = None,
    test_dir: str = None
    num_workers: int = 8
    batch_size: int = 256
    seed: int = 1
    max_epochs: int = 20
    input_size: int = 128 # input resolution
    num_ftrs: int = 32
    accelerator: str = "cuda" if torch.cuda.is_available() else "cpu"
    devices: bool = 1
    stage: str = "train"
'''
# set the seed
pl.seed_everything(seed)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)
'''

class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init()
        resnet = torchvision.models.resnet18