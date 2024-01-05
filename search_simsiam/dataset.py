import glob
import pandas as pd
import sys
sys.path.append('/home/subhamoy/search/hits-sdo-similaritysearch/')
from torch.utils.data import Dataset
import numpy as np
from sdo_augmentation.augmentation import Augmentations
from sdo_augmentation.augmentation_list import AugmentationList
from search_utils.image_utils import read_image
import torch
import random

class HMItilesDataset(Dataset):
    '''
        Dataset to load jpg patches produced by the search packager:
        https://github.com/hits-sdo/hits-sdo-packager
        
        and downloaded by the downloader:
        https://github.com/hits-sdo/hits-sdo-downloader
        
        It performs single or double augmentation on each patch using
        the augmentation modules of the packager
    '''
    def __init__(self, data_path: str, augmentation: str='single',
                 data_stride:int = 1, datatype=np.float32,
                 offlimb_frac = 1, step_multiplier:int = 1,
                 batch_size:int = 64, instr: str='mag'):
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
        random.seed(23)
        self.data_path = data_path
        
        if instr.lower()=='mag':
            df = pd.read_csv('/d0/subhamoy/models/search/magnetograms/indexer.csv')
            df = df[df['offlimb_frac_area']<=offlimb_frac].reset_index(drop=True)
            f_list = sorted(list(df['filename']))
        else:
            f_list = sorted(glob.glob(data_path + "/**/*.jpg", recursive=True))
            
        random.shuffle(f_list)
        f_list = f_list[:batch_size*(len(f_list)//batch_size)]
        f_final = f_list.copy()
        for n in range(step_multiplier-1):
            random.seed(23 + n + 1)
            f_suppl = f_list.copy()
            random.shuffle(f_suppl)
            f_final = f_final + f_suppl 
            
        #self.image_files = f_list * step_multiplier
        self.image_files = f_final
            
        if data_stride>1:
            self.image_files = self.image_files[::data_stride]
        self.augmentation_list = AugmentationList(instrument="euv")
        #self.augmentation_list.keys.remove('brighten')
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
        image = read_image(image_loc = self.image_files[idx],
                           image_format="jpg")

        if self.augmentation.lower() != 'none':
            # image2 = image.copy()

            aug = Augmentations(image, self.augmentation_list.randomize())
            image2, _ = aug.perform_augmentations(fill_void='Nearest')

            if self.augmentation.lower() == 'double':
                aug = Augmentations(image, self.augmentation_list.randomize())
                image, _ = aug.perform_augmentations(fill_void='Nearest')

            image = np.moveaxis(image, [0, 1, 2], [1, 2, 0]).astype(self.datatype)           
            image2 = np.moveaxis(image2, [0, 1, 2], [1, 2, 0]).astype(self.datatype)           

            return image, image2, self.image_files[idx]

        else:

            image = np.moveaxis(image, [0, 1, 2], [1, 2, 0]).astype(self.datatype)           
            return image, self.image_files[idx]
        
if __name__ == '__main__':
    DATA_DIR = '/d0/euv/aia/preprocessed/HMI/HMI_256x256/'
    dataset_train_simsiam = HMItilesDataset(data_path=DATA_DIR, augmentation='double',
                                               data_stride=1, offlimb_frac=1)
    
    dataloader_train_simsiam = torch.utils.data.DataLoader(
        dataset_train_simsiam,
        batch_size=64,
        shuffle=False,                                                                                                                 
        drop_last=True,                                                                 
        num_workers=4)
    print(dataloader_train_simsiam.image_files[:10])
    im1, im2, f = next(iter(dataloader_train_simsiam))
    print(f[:10])
    