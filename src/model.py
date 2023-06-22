import sys,os
sys.path.append(os.getcwd())

import copy
from lightly.loss import NegativeCosineSimilarity, NTXentLoss
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead, SimSiamPredictionHead, SimSiamProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
import numpy as np
from torch import nn, optim
import torch
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class SimSiam(nn.Module):
    def __init__(self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(num_ftrs, proj_hidden_dim, out_dim)
        self.prediction_head = SimSiamPredictionHead(out_dim, pred_hidden_dim, out_dim)

    def forward(self, x):
        # get representations
        f = self.backbone(x).flatten(start_dim=1)
        # get projections
        z = self.projection_head(f)
        # get predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()
        return z, p


class BYOL(pl.LightningModule):
    """
        PyTorch Lightning module for self supervision BYOL model

        Parameters:
            lr (float):                 learning rate
            wd (float):                 L2 regularization parameter
            epochs (int):               Number of epochs for scheduler
    """
    def __init__(self, lr=0.1, wd=1e-3, input_channels=1, projection_size=2, prediction_size=2, 
                 cosine_scheduler_start=0.9, cosine_scheduler_end=1.0, epochs=10, loss='contrast'):
        super().__init__()

        resnet = torchvision.models.resnet18()
        # change number of input channels 
        resnet.conv1 = nn.Conv2d(input_channels,64,kernel_size=(7,7),stride=(2,2),padding=(1,1),bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(512, 1024, projection_size)
        self.prediction_head = BYOLPredictionHead(projection_size, 1024, prediction_size)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # define loss function
        if loss == 'contrast':
            self.loss = NTXentLoss()
        else:
            self.loss = NegativeCosineSimilarity()

        self.cosine_scheduler_start = cosine_scheduler_start
        self.cosine_scheduler_end = cosine_scheduler_end
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = wd
        self.projection_size=projection_size

        self.save_hyperparameters()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
    
    def embed(self,x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        return y

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.epochs, self.cosine_scheduler_start, self.cosine_scheduler_end)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
       
        _,x0, x1 = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)

        loss = 0.5 * (self.loss(p0, z1) + self.loss(p1, z0))
        # loss = 0.5 * (self.loss_contrast(p0, z1) + self.loss_contrast(p1, z0))

        self.log('loss', loss)
        return loss

    def validation_step(self,batch,batch_idx):
        """
            Runs the model on the validation set and logs validation loss 
            and other metrics.

            Parameters:
                batch:                  batch from a DataLoader
                batch_idx:              index of batch                  
        """
        _,x0,x1 = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)

        val_loss = 0.5*(self.loss(p0,z1)+self.loss(p1,z0))

        # calculate the per-dimension standard deviation of the outputs
        # we can use this later to check whether the embeddings are collapsing
        output = self.embed(x0)
        output = F.normalize(output, dim=1)
        output_std = output.std(dim=0)
        output_std = output_std.mean()
        collapse_level = max(0.0,1-np.sqrt(512)*output_std)

        self.log_dict({'val_loss':val_loss,
                       'collapse_level':collapse_level},
                      on_step=False,on_epoch=True)

    def test_step(self,batch,batch_idx):
        """
            Runs the model on the test set and logs test metrics 

            Parameters:
                batch:                  batch from a DataLoader
                batch_idx:              index of batch                  
        """
        pass

    def configure_optimizers(self):
        """
            Sets up the optimizer and learning rate scheduler.
            
            Returns:
                optimizer:              A torch optimizer
        """
        # optimizer = optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.epochs)
        
        return [optimizer],[scheduler]

    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        """
            Forward pass of model for prediction

            Parameters:
                batch:          batch from a DataLoader
                batch_idx:      batch index
                dataloader_idx

            Returns:
                embedding:      embeddings for batch
        """
        f,x0,_ = batch
        embedding = self.embed(x0)
        embedding_proj = self.forward_momentum(x0)
        return f,embedding,embedding_proj
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)