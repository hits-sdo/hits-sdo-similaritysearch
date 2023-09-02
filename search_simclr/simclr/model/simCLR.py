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

import math


class SimCLR(pl.LightningModule):
    def __init__(self, lr, model_str = 'resnet18', output_dim=128):
        super().__init__()
        model_str = model_str.lower()
        if model_str == 'resnet18':
            feature_extractor = torchvision.models.resnet18()
        elif model_str == 'resnet34':
            feature_extractor = torchvision.models.resnet34()
        elif model_str == 'resnet50':
            feature_extractor = torchvision.models.resnet50()
        elif model_str == 'resnet101':
            feature_extractor = torchvision.models.resnet101()
        elif model_str == 'resnet152':
            feature_extractor = torchvision.models.resnet152()
        elif model_str == 'densenet121':
            feature_extractor = torchvision.models.densenet121()
            
        self.backbone = nn.Sequential(*list(feature_extractor.children())[:-1])
        hidden_dim = feature_extractor.fc.in_features

        self.projection_head = SimCLRProjectionHead(input_dim = hidden_dim, hidden_dim = hidden_dim, output_dim=output_dim)
        # SimCLRProjectionHead(input_dim: int = 2048, hidden_dim: int = 2048, 
        # output_dim: int = 128, num_layers: int = 2, batch_norm: bool = True)
        self.criterion = NTXentLoss()
        self.lr = lr
        
    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z
    
    def training_step(self, batch, batch_idx):
        x0, x1, _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        output = torch.nn.functional.normalize(z0.detach(), dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        collapse = max(0.0, 1 - math.sqrt(self.out_dim) * output_std)
        wandb.log({"train_loss_ssl": loss})
        wandb.log({"collapse_level": collapse})
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
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