from matplotlib.pyplot import imshow, figure
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from lightly.data import LightlyDataset
from dataset import AIAHMItilesDataset
import torch
import wandb
import torchvision
from embedding_utils import EmbeddingUtils
from gui_utils import fetch_n_neighbor_filenames
from PIL import Image
import glob
from model import load_model
class SimSiamPlotter(pl.Callback):
    def __init__(self, stride = 1, instr = 'mag'):
        super().__init__()
        self.stride = stride
        self.instr = instr

    def on_train_epoch_end(self, trainer, pl_module):
        if self.instr.lower()=='mag':
            query_image = Image.open('/home/subhamoy/search/latest_4096_HMIBC.jpg')
            DATA_DIR = '/d0/euv/aia/preprocessed/HMI/HMI_256x256/'
        else:
            query_image = Image.open('/d0/euv/aia/preprocessed/AIA_211_193_171/raw/20230714_074800_aia_211_193_171.jpg')
            DATA_DIR = '/d0/euv/aia/preprocessed/AIA_211_193_171/AIA_211_193_171_256x256/'
        
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        dataset_train_simsiam = AIAHMItilesDataset(data_path=DATA_DIR,
                                                augmentation=None, 
                                                instr=self.instr,
                                                data_stride=self.stride)
        
        pl_module.eval()
        E = EmbeddingUtils(model=pl_module,
                           dataset=dataset_train_simsiam,
                           batch_size=64,
                           num_workers=4)
        
        filenames, embeddings = E.embedder()
        
        dct = {'filenames':filenames, 'embeddings': embeddings}
        
        tensor = transforms(query_image)
        
        ys_ = 1500
        if self.instr.lower()=='mag':
            xs_ = 2400
        else:
            xs_ = 1100
        tensor = tensor[None, :3, ys_:(ys_ + 256), xs_:(xs_ + 256)]
        
        with torch.no_grad():
            embedding = pl_module.backbone(tensor.to('cuda')).flatten(start_dim=1)
        query_embedding = embedding[0].cpu().data.numpy()
        filenames = fetch_n_neighbor_filenames(query_embedding,
                                            dct,
                                            "EUCLIDEAN",
                                            num_images=4)
        images = [np.array(query_image), np.array(query_image)[ys_:(ys_ + 256), xs_:(xs_ + 256),:]]
        titles = ['Full-disk','Query RoI']
        for i,f in enumerate(filenames):
            images.append(np.array(Image.open(f)))
            n = i+1
            titles.append(f"NN {n}")

 
        fig, axes = plt.subplots(1,6, figsize=(24,4), constrained_layout=True)
        ax = axes.ravel()

        for i, t in enumerate(titles):
            ax[i].imshow(images[i])
            if i==0:
                xs = [xs_, xs_ + 256, xs_ + 256, xs_, xs_]
                ys = [ys_, ys_, ys_ + 256, ys_ + 256, ys_]
                ax[i].plot(xs, ys, color="red", linewidth=2.0)
            ax[i].set_title(t)
                
        pl_module.train()
        wandb.log({"Retrieved Images": wandb.Image(fig)})
        