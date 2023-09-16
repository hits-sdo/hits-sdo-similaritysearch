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
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import wandb


from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

import math


class SimCLR(pl.LightningModule):
    def __init__(self, lr = 0.02, model_str = 'resnet18', output_dim=128):
        super().__init__()
        model_str = model_str.lower()
        if model_str == 'resnet18':
            feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_str == 'resnet34':
            feature_extractor = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif model_str == 'resnet50':
            feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_str == 'resnet101':
            feature_extractor = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif model_str == 'resnet152':
            feature_extractor = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        elif model_str == 'densenet121':
            feature_extractor = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            
        self.backbone = nn.Sequential(*list(feature_extractor.children())[:-1])
        hidden_dim = feature_extractor.fc.in_features

        self.projection_head = SimCLRProjectionHead(input_dim = hidden_dim, hidden_dim = hidden_dim, output_dim=output_dim)
        # SimCLRProjectionHead(input_dim: int = 2048, hidden_dim: int = 2048, 
        # output_dim: int = 128, num_layers: int = 2, batch_norm: bool = True)
        self.out_dim = output_dim
        self.criterion = NTXentLoss()
        self.lr = lr
        self.save_hyperparameters() # Saves hyperparameters so that when we load from a checkpoint, we can use the same hyperparameters
        
    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        # Save the output of the base encoder as base_encoder_output
        self.base_encoder_output = h
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
        collapse = 1 - math.sqrt(self.out_dim) * output_std
        wandb.log({"train_loss_ssl": loss})
        wandb.log({"collapse_level": collapse})
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return optim #[optim], [scheduler]
    
class SimCLRCallback(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path

    def on_after_backward(self, trainer, pl_module):
        # Access the embeddings at the output of the base encoder
        embeddings = pl_module.base_encoder_output

        # Save the embeddings to a file
        torch.save(embeddings, self.save_path)


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