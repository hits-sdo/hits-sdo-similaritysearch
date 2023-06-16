import sys,os
sys.path.append(os.getcwd())

import glob
import torch
from torchvision import transforms
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from search_utils.image_utils import read_image
from torch.utils.data import Dataset,DataLoader
from sdo_augmentation.augmentation import Augmentations
from sdo_augmentation.augmentation_list import AugmentationList

class TilesDataset(Dataset):
    """
        Pytorch dataset for handling magnetogram tile data 
        
    """
    def __init__(self, data_path: str, augmentation: str='single',
                 instrument:str='mag',filetype:str='npy',
                 datatype=np.float32):
        '''
            Initializes image files in the dataset
            
            Args:
                data_path (str): path to the folder containing the images
                augmentation (str): whether to just return the original patches ('none')
                                    perform single augmentation ('single') 
                                    or perform double augmentation ('double').
                                    No augmentation returns a single image,
                                    single or double augmentation returns two.
                data_stride (int): stride to use when loading the images to work 
                                    with a reduced version of the data
                datatype (numpy.dtype): datatype to use for the images
        '''
        self.data_path = data_path
        self.image_files = glob.glob(data_path + "/**/*."+filetype, recursive=True)
        self.augmentation_list = AugmentationList(instrument=instrument)
        self.filetype=filetype
        self.datatype=datatype
        self.augmentation = augmentation
        if self.augmentation is None:
            self.augmentation = 'none'

    def __len__(self):
        '''
            Calculates the number of images in the dataset
                
            Returns:
                int: number of images in the dataset
        '''
        return len(self.image_files)

    def __getitem__(self, idx):
        '''
            Retrieves an image from the dataset and creates a copy of it,
            applying a series of random augmentations to the copy.

            Args:
                idx (int): index of the image to retrieve
                
            Returns:
                tuple: (image, image2) where image2 is an augmented modification
                of the input and image can be the original image, or another augmented
                modification, in which case image2 is double augmented
        '''
        file = self.image_files[idx]
        image = read_image(image_loc=file,image_format="npy")
        
        # Normalize magnetogram data
        # clip magnetogram data within max value
        maxval = 1000  # Gauss
        image[np.where(image>maxval)] = maxval
        image[np.where(image<-maxval)] = -maxval
        # scale between -1 and 1
        image = (image+maxval)/2/maxval

        image2 = image.copy()

        if self.augmentation.lower() != 'none':

            aug = Augmentations(image, self.augmentation_list.randomize())
            image2, _ = aug.perform_augmentations(fill_void='Nearest')

            if self.augmentation.lower() == 'double':
                aug = Augmentations(image, self.augmentation_list.randomize())
                image, _ = aug.perform_augmentations(fill_void='Nearest')

        if image.ndim == 3:
            image = np.moveaxis(image, [0, 1, 2], [1, 2, 0]).astype(self.datatype)           
            image2 = np.moveaxis(image2, [0, 1, 2], [1, 2, 0]).astype(self.datatype)           
        elif image.ndim == 2:
            image = np.expand_dims(image,0)
            image2 = np.expand_dims(image2,0)

        return file,torch.Tensor(image), torch.Tensor(image2)

    
class TilesDataModule(pl.LightningDataModule):
    """
    Datamodule for self supervision on tiles dataset
    """

    def __init__(self,data_path:str,batch:int=128,augmentation:str='double'):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch
        self.augmentation = augmentation

    def prepare_data(self):
        pass

    def setup(self,stage:str):
        self.train_set = TilesDataset(self.data_path,self.augmentation)
        self.data_set = TilesDataset(self.data_path,augmentation='none')

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=4,shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.data_set,batch_size=self.batch_size,num_workers=4)