import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import copy

from lightly.loss import NegativeCosineSimilarity, NTXentLoss
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

class BYOL(pl.LightningModule):
    def __init__(self, lr=0.1, projection_size=256, prediction_size=256, cosine_scheduler_start=0.1, cosine_scheduler_end=1.0, epochs=10, loss='cos'):
        super().__init__()

        resnet = torchvision.models.resnet18() # Play w/ resnet.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(512, 1024, projection_size)
        self.prediction_head = BYOLPredictionHead(projection_size, 1024, prediction_size)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)


        self.loss = loss
        self.loss_cos = NegativeCosineSimilarity()
        self.loss_contrast = NTXentLoss()

        self.cosine_scheduler_start = cosine_scheduler_start
        self.cosine_scheduler_end = cosine_scheduler_end
        self.epochs = epochs
        self.lr = lr

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

    def training_step(self, batch, batch_idx):

        momentum = cosine_schedule(self.current_epoch, self.epochs, self.cosine_scheduler_start, self.cosine_scheduler_end)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1, _) = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)

        loss_cos = 0.5 * (self.loss_cos(p0, z1) + self.loss_cos(p1, z0))
        loss_contrast = 0.5 * (self.loss_contrast(p0, z1) + self.loss_contrast(p1, z0))

        if self.loss == 'cos':
            loss = loss_cos
        else:
            loss = loss_contrast

        self.log('loss cos', loss_cos)
        self.log('loss contrast', loss_contrast)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr) # Play w/ optimizers.
