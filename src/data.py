import sys,os
sys.path.append(os.getcwd())

import glob
import torch
import torchvision.transforms.v2 as transforms
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
from search_utils.image_utils import read_image
from torch.utils.data import Dataset,DataLoader
from sdo_augmentation.augmentation import Augmentations
from sdo_augmentation.augmentation_list import AugmentationList

class TilesDataset(Dataset):
    """
        Pytorch dataset for handling magnetogram tile data 
        
    """
    def __init__(self, image_files: list, transform:transforms.ToTensor(), augmentation: str='single',
                 instrument:str='mag',filetype:str='npy',
                 datatype=np.float32):
        '''
            Initializes image files in the dataset
            
            Args:
                image_files (list): list of file paths to images
                augmentation (str): whether to just return the original patches ('none')
                                    perform single augmentation ('single') 
                                    or perform double augmentation ('double').
                                    No augmentation returns a single image,
                                    single or double augmentation returns two.
                data_stride (int): stride to use when loading the images to work 
                                    with a reduced version of the data
                datatype (numpy.dtype): datatype to use for the images
        '''
        self.image_files = image_files
        self.transform = transform
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
        image = read_image(image_loc=file,image_format=self.filetype)
        
        # Normalize magnetogram data
        # clip magnetogram data within max value
        maxval = 1000  # Gauss
        image[np.where(image>maxval)] = maxval
        image[np.where(image<-maxval)] = -maxval
        # scale between -1 and 1
        image = (image+maxval)/2/maxval
        image = np.expand_dims(image,0)

        image2 = image.copy()
        image = torch.Tensor(image)
        image2 = torch.Tensor(image2)

        if self.augmentation.lower() != 'none':

            # aug = Augmentations(image, self.augmentation_list.randomize())
            # image2, _ = aug.perform_augmentations(fill_void='Nearest')
            image2 = self.transform(image)

            if self.augmentation.lower() == 'double':
                image = self.transform(image)    

        return file,image,image2

    
class TilesDataModule(pl.LightningDataModule):
    """
    Datamodule for self supervision on tiles dataset
    """

    def __init__(self,data_path:str,batch:int=128,augmentation:str='double',filetype:str='npy',dim:int=128):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch
        self.augmentation = augmentation
        self.filetype = filetype

        # define data transforms - augmentation for training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomInvert(p=0.3),
            transforms.RandomAdjustSharpness(1.5,p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([
                 transforms.GaussianBlur(kernel_size=9,sigma=(0.1,4.0)),
                 ]),p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([
                 transforms.RandomRotation(degrees=45,fill=0.5)
                 ]),p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ScaleJitter(target_size=(128,128),scale_range=(0.7,1.3),antialias=True),
            transforms.Resize((dim,dim),antialias=True)
        ])

    def prepare_data(self):
        self.image_files = glob.glob(self.data_path + "/**/*."+self.filetype, recursive=True)

    def setup(self,stage:str):
        # split into training and validation
        random.shuffle(self.image_files)
        train_files = self.image_files[:int(0.8*len(self.image_files))]
        val_files = self.image_files[int(0.8*len(self.image_files)):]
        self.train_set = TilesDataset(train_files,self.transform,self.augmentation)
        self.val_set = TilesDataset(val_files,self.transform,augmentation='single')
        self.trainval_set = TilesDataset(self.image_files,augmentation='none')

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=4,shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.trainval_set,batch_size=self.batch_size,num_workers=4)