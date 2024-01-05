import math
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

from lightly.loss import NegativeCosineSimilarity, NTXentLoss
from lightly.models.modules.heads import (
    SimSiamPredictionHead,
    SimSiamProjectionHead
)


# LightingModule
class SimSiam(pl.LightningModule):
    def __init__(self, num_ftrs=512, proj_hidden_dim=512, pred_hidden_dim=128, out_dim=512,
                 lr=0.0125, lr_schedule_coeff = 1, optim_type = 'sgd', pretrained = False,
                 contrastive = False):
        super().__init__()
        if pretrained:
            resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        else:
            resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimSiamProjectionHead(num_ftrs, proj_hidden_dim, out_dim)
        self.prediction_head = SimSiamPredictionHead(out_dim, pred_hidden_dim, out_dim)
        if contrastive:
            self.criterion = NTXentLoss()
        else:
            self.criterion = NegativeCosineSimilarity()
        self.out_dim = out_dim
        self.lr = lr
        self.lr_schedule_coeff = lr_schedule_coeff
        self.optim_type = optim_type

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        # (x0, x1) = batch[0]
        x0, x1, f = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        output = torch.nn.functional.normalize(p0.detach(), dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        collapse = max(0.0, 1 - math.sqrt(self.out_dim) * output_std)
        self.log('loss', loss)
        self.log('collapse_level', collapse)
        return loss

    def configure_optimizers(self):
        
        if self.optim_type=='sgd':
            optimizer = torch.optim.SGD([{
                'params': [p for p in self.parameters()],
                'name': 'learning_rate'}], lr=self.lr)#, weight_decay=0.1)
        else:
            optimizer = torch.optim.Adam([{
                'params': [p for p in self.parameters()],
                'name': 'learning_rate'}], lr=self.lr)#, weight_decay=0.1)  
        lambda1 = lambda epoch: self.lr_schedule_coeff ** epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        return [optimizer], [lr_scheduler]
    
        #optim = torch.optim.SGD(self.parameters(), lr=self.lr)
        #return optim


def load_model(model_path):
    model = SimSiam.load_from_checkpoint(model_path)
    return model
