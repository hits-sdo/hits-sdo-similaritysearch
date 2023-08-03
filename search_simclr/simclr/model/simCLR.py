import pyprojroot
import os

root = pyprojroot.here()
utils_dir = root/'search_utils'

import sys
sys.path.append(str(root))
from search_simclr.simclr.dataloader.dataset import SdoDataset
from search_utils import image_utils  # TODO needed?
from search_simclr.simclr.dataloader.dataset_aug import Transforms_SimCLR
from search_simclr.simclr.dataloader.datamodule import SimCLRDataModule

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import wandb


from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()
        
    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z
    
    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        wandb.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return optim #[optim], [scheduler]

# note on line 47:
'''
In the context of the code snippet from zablo.net, 
the SimCLRProjectionHead class is a part of the SimCLR neural 
network for embeddings. 
It is used to add a projection head on top of the base model's 
output to further process the image embeddings.

The SimCLRProjectionHead class takes three arguments: 512
as the input size, 512 as the hidden size, and 128 as the
output size. These dimensions determine the shape and size 
of the linear layers used in the projection head.'''