import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import os
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
                 train_flist: str = None, # os.path.join(root, "data", "miniset", "AIA171", "train_val_simclr", "train_file_list.txt"),
                 val_flist: str = None, # os.path.join(root, "data", "miniset", "AIA171", "train_val_simclr", "val_file_list.txt"),
                 # train_flist: str, val_flist, test_flist ...etc 
                 test_flist: str = None):
         
        super().__init__()
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.train_flist = train_flist
        self.val_flist = val_flist
        self.test_flist = test_flist

        # TODO: want to make this a variable to pass in instead of hardcoding
        # another note: want truly random values to be passed into these?
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
                                                file_list=self.train_flist,
                                                transform=self.transform
                                                )

            case "val":
                self.val_dataset = SdoDataset(tile_dir=self.val_dir,
                                              file_list=self.val_flist,
                                              transform=self.transform
                    )

            case "test":
                self.test_dataset = SdoDataset(tile_dir=self.test_dir,
                                               file_list=self.test_flist,
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
    train_dir = os.pathroot/"data/miniset/AIA171/monochrome"
    train_list = os.path.join(root, "data", "miniset", "AIA171", "train_val_simclr", "train_file_list.txt")
    val_list = os.path.join(root, "data", "miniset", "AIA171", "train_val_simclr", "val_file_list.txt")
    simclr_dm = SimCLRDataModule()
    simclr_dm.setup("train")

    for batch_idx, (img1, img2, fname, _) in enumerate(simclr_dm.train_dataloader()):
        print (batch_idx, img1.shape, img2.shape, fname)
        break


if __name__ == "__main__":
    main()