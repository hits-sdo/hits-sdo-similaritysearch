'''
    Script with the tile dataset used in the dataloader of the
    search-byol implementation
'''

import glob
from torch.utils.data import Dataset
import numpy as np
from sdo_augmentation.augmentation import Augmentations
from sdo_augmentation.augmentation_list import AugmentationList
from search_utils.image_utils import read_image

class SDOTilesDataset(Dataset):
    '''
        Dataset to load jpg patches produced by the search packager:
        https://github.com/hits-sdo/hits-sdo-packager
        
        and downloaded by the downloader:
        https://github.com/hits-sdo/hits-sdo-downloader
        
        It performs single or double augmentation on each patch using
        the augmentation modules of the packager
    '''
    def __init__(self, data_path: str, augmentation: str='single',
                 data_stride:int = 1, datatype=np.float32):
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
        self.image_files = glob.glob(data_path + "/**/*.jpg", recursive=True)
        if data_stride>1:
            self.image_files = self.image_files[::data_stride]
        self.augmentation_list = AugmentationList(instrument="euv")
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
            image2 = image.copy()

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
