import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

import pyprojroot

root = pyprojroot.here()
utils_dir = root/'search_utils'

import sys
sys.path.append(str(root))
from search_simclr.simclr.dataloader.dataset import SdoDataset
from search_utils import image_utils  # TODO needed?
from search_simclr.simclr.dataloader.dataset_aug import Transforms_SimCLR


class SimCLRDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 train_dir: str = root/"data/miniset/AIA171/monochrome",
                 val_dir: str = None,
                 test_dir: str = None):
         
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.transform = Transforms_SimCLR(blur=(1,1), 
                                              brighten=1.0, 
                                              translate=(1, 1), 
                                              zoom=1.0, 
                                              rotate=45.0, 
                                              noise_mean=0.0, 
                                              noise_std=0.05)
    
    def prepare_data(self):
        #TODO download call team red dataloader
        pass

    def setup(self, stage: str):
        match stage:
            case "train":
                self.train_dataset = SdoDataset(tile_dir=self.train_dir, 
                                        transform=self.transform
                                        )

            case "val":
                self.val_dataset = SdoDataset(tile_dir=self.val_dir,
                    transform=self.transform
                    )

            case "test":
                self.test_dataset = SdoDataset(tile_dir=self.test_dir,
                    transform=self.transform
                    )

    def train_dataloader(self): # Return DataLoader for Training Data
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True)

    def val_dataloader(self): # Return DataLoader for Validation Data
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True)

    def test_dataloader(self): # Return DataLoader for Testing Data
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True)

def main():
    train_dir = root/"data/miniset/AIA171/monochrome"
    simclr_dm = SimCLRDataModule()
    simclr_dm.setup("train")

    for batch_idx, (img1, img2, fname, _) in enumerate(simclr_dm.train_dataloader()):
        print (batch_idx, img1.shape, img2.shape, fname)
        break


if __name__ == "__main__":
    main()