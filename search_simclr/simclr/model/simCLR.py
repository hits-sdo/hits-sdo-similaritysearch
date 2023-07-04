import pyprojroot
import os

root = pyprojroot.here()
utils_dir = root/'search_utils'

import sys
sys.path.append(str(root))
from search_simclr.simclr.dataloader.dataset import SdoDataset
from search_utils import image_utils  # TODO needed?
from search_simclr.simclr.dataloader.dataset_aug import Transforms_SimCLR
# from search_simclr.simclr.dataloader import dataset_aug
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
# 
# from lightly.data import LightlyDataset
# from lightly.transforms import SimCLRTransform, utils

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

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
    
def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


# move this main function code to a script
def main():
    config = SDOConfig()
    pl.seed_everything(config.seed)
    sdo_datamodule = SimCLRDataModule(batch_size=config.batch_size, train_dir=config.train_dir,val_dir=config.val_dir,test_dir=config.test_dir)
    sdo_datamodule.setup(stage=config.stage)
    # 

    #accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    model = SimCLR()
    trainer = pl.Trainer(max_epochs=config.max_epochs, devices=config.devices, accelerator=config.accelerator)
    trainer.fit(model, sdo_datamodule.train_dataloader())
    
    # trainer.fit(model, dataloader_train_simclr)

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