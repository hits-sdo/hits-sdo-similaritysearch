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
from search_utils.file_utils import get_file_list


class SimCLRDataModule(pl.LightningDataModule):
    def __init__(self,
                 blur: tuple(int, int) = (5,5), 
                 brighten: float = 1.0, 
                 translate: tuple(int, int) =(1, 3), 
                 zoom: float = 1.5, 
                 rotate: float = 360.0, 
                 noise_mean: float = 0.0, 
                 noise_std: float = 0.05, 
                 cutout_holes: int = 1, 
                 cutout_size: float = 0.3,
                 batch_size: int = 32,
                 
                 tile_dir: str = None,
                 train_dir: str = None,
                 val_dir: str = None,
                 test_dir: str = None,
                 
                 train_flist: str = None, 
                 val_flist: str = None, 
                 test_flist: str = None,
                 tot_fpath_wfname: str = None,
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
        self.train_flist = train_flist
        self.val_flist = val_flist
        self.tot_fpath_wfname = tot_fpath_wfname
        self.batch_size = batch_size
        
        # train_val_dir = os.path.join(root , 'data', 'train_val_simclr')
        # self.val_flist = os.path.join(train_val_dir, 'val_file_list.txt')
        # self.train_flist = os.path.join(train_val_dir,'train_file_list.txt')
        self.test_flist = test_flist
        #tile_dir = os.path.join(root , 'data')
        # self.tot_fpath_wfname = os.path.join(root , 'data' , 'train_val_simclr', 'tot_full_path_files.txt')
        #tile_dir.replace(os.sep, "/")

        # TODO: want to make this a variable to pass in instead of hardcoding
        # another note: want truly random values to be passed into these?
        
    
    def prepare_data(self):
        #TODO download call team red dataloader
        # data\AIA211_193_171_Miniset\20100601_000008_aia_211_193_171\tiles
        
        #train_val_dir.replace(os.sep, "/")
        # Todo: FIXME !!!! Make tot_file_list a list of file full paths, not just file names
        # Todo: Use get_file_list_from_dir_recrusive() from search_utils/file_utils.py
        tot_file_list = get_file_list(self.tot_fpath_wfname)
        train_file_list, val_file_list = partition_tile_dir_train_val(tot_file_list[:50], 0.8)
        # save lists

        with open(self.train_flist, 'w') as f:
            for item in train_file_list:
                f.write("%s\n" % item)
        with open(self.val_flist, 'w') as f:
            for item in val_file_list:
                f.write("%s\n" % item)

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
                self.train_dataset = SdoDataset(tile_dir=self.train_dir, 
                                                file_list=self.train_flist,
                                                transform=transform
                                                )

            case "val":
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
                self.val_dataset = SdoDataset(tile_dir=self.val_dir,
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
                self.test_dataset = SdoDataset(tile_dir=self.test_dir,
                                               file_list=self.test_flist,
                                               transform=transform
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