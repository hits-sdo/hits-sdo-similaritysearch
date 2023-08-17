import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pyprojroot

root = pyprojroot.here()
utils_dir = root/'search_utils'

import sys
sys.path.append(str(root))
from search_simclr.simclr.dataloader.dataset import SdoDataset, partition_tile_dir_train_val
from search_utils import image_utils  # TODO needed?
from search_simclr.simclr.dataloader.dataset_aug import Transforms_SimCLR
from search_utils.file_utils import get_file_list, split_val_files
from typing import Tuple



class SimCLRDataModule(pl.LightningDataModule):
    def __init__(self,
                 blur: Tuple[int, int] = Tuple[5,5], 
                 brighten: float = 1.0, 
                 translate: Tuple[int, int] = Tuple[1, 3], 
                 zoom: float = 1.5, 
                 rotate: float = 360.0, 
                 noise_mean: float = 0.0, 
                 noise_std: float = 0.05, 
                 cutout_holes: int = 1, 
                 cutout_size: float = 0.3,
                 batch_size: int = 32,
                 num_images: int = None,
                 percent: float = 0.8,
                 
                 tile_dir: str = None,
                 train_dir: str = None,
                 val_dir: str = None,
                 test_dir: str = None,
                 
                 train_fpath: str = None,
                 val_fpath: str = None,
                 train_flist: str = None, 
                 val_flist: str = None, 
                 test_flist: str = None,
                 tot_fpath_wfname: str = None,
                 split: bool = False,
                 num_workers: int = 1
                 ):
         
        super().__init__()
        # Define Augmentations
        self.blur = blur 
        self.brighten = brighten
        self.translate = translate
        self.zoom = zoom
        self.rotate = rotate
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.cutout_holes = cutout_holes
        self.cutout_size = cutout_size
        
        
        # Define Datasets
        self.tile_dir = tile_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.train_fpath = train_fpath
        self.val_fpath = val_fpath
        self.train_flist = train_flist
        self.val_flist = val_flist
        self.tot_fpath_wfname = tot_fpath_wfname
        self.batch_size = batch_size
        self.num_images = num_images
        self.percent = percent
        self.split = split
        
        # train_val_dir = os.path.join(root , 'data', 'train_val_simclr')
        # self.val_flist = os.path.join(train_val_dir, 'val_file_list.txt')
        # self.train_flist = os.path.join(train_val_dir,'train_file_list.txt')
        self.test_flist = test_flist
        self.num_workers = num_workers
        #tile_dir = os.path.join(root , 'data')
        # self.tot_fpath_wfname = os.path.join(root , 'data' , 'train_val_simclr', 'tot_full_path_files.txt')
        #tile_dir.replace(os.sep, "/")

        # TODO: want to make this a variable to pass in instead of hardcoding
        # another note: want truly random values to be passed into these?
        
    
    def prepare_data(self):
        if self.split or not (os.path.exists(self.train_fpath) and os.path.exists(self.val_fpath)):
            split_val_files(self.tot_fpath_wfname, self.train_fpath, self.val_fpath, self.num_img, self.percent)
        self.train_flist = get_file_list(self.train_fpath)
        self.val_flist = get_file_list(self.val_fpath)
        print("in prepare_data")       
        

    def setup(self, stage: str):
        match stage:
            case "train":
                transform = Transforms_SimCLR(blur=self.blur, 
                                              brighten=self.brighten, 
                                              translate=self.translate, 
                                              zoom=self.zoom, 
                                              rotate=self.rotate, 
                                              noise_mean=self.noise_mean, 
                                              noise_std=self.noise_std, 
                                              cutout_holes=self.cutout_holes, 
                                              cutout_size=self.cutout_size,
                                              data_dir=self.tile_dir,
                                              file_list=self.train_dir
                                              )
                self.train_dataset = SdoDataset(tile_dir=self.tile_dir, 
                                                file_list=self.train_flist,
                                                transform=transform
                                                )

            case "validate":
                transform = Transforms_SimCLR(blur=self.blur, 
                                              brighten=self.brighten, 
                                              translate=self.translate, 
                                              zoom=self.zoom, 
                                              rotate=self.rotate, 
                                              noise_mean=self.noise_mean, 
                                              noise_std=self.noise_std, 
                                              cutout_holes=self.cutout_holes, 
                                              cutout_size=self.cutout_size,
                                              data_dir=self.tile_dir,
                                              file_list=self.val_dir
                                            )
                self.val_dataset = SdoDataset(tile_dir=self.tile_dir,
                                              file_list=self.val_flist,
                                              transform=transform
                    )

            case "test":
                transform = Transforms_SimCLR(blur=self.blur, 
                                              brighten=self.brighten, 
                                              translate=self.translate, 
                                              zoom=self.zoom, 
                                              rotate=self.rotate, 
                                              noise_mean=self.noise_mean, 
                                              noise_std=self.noise_std, 
                                              cutout_holes=self.cutout_holes, 
                                              cutout_size=self.cutout_size,
                                              data_dir=self.tile_dir,
                                              file_list=self.test_dir
                                            )
                self.test_dataset = SdoDataset(tile_dir=self.tile_dir,
                                               file_list=self.test_flist,
                                               transform=transform
                    )
                

    def train_dataloader(self): # Return DataLoader for Training Data
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self): # Return DataLoader for Validation Data
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self): # Return DataLoader for Testing Data
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
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