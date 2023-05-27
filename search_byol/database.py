import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
import glob
import numpy as np
from sdo_augmentation.augmentation import Augmentations
from sdo_augmentation.augmentation_list import AugmentationList
from search_utils.image_utils import read_image

class SDOTilesDataset(Dataset):
    def __init__(self, data_path: str, double_augmentation: bool, data_stride:int = 1):
        self.data_path = data_path
        self.image_files = glob.glob(data_path + "/**/*.jpg", recursive=True)
        if data_stride>1:
            self.image_files = self.image_files[::data_stride]
        self.augmentation_list = AugmentationList(instrument="euv")
        self.double_augmentation = double_augmentation

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = read_image(image_loc = self.image_files[idx],
                           image_format="jpg")
        image2 = image.copy()

        a = Augmentations(image, self.augmentation_list.randomize())
        image2, _ = a.perform_augmentations(fill_void='Nearest')
    
        if self.double_augmentation:
            a = Augmentations(image, self.augmentation_list.randomize())
            image, _ = a.perform_augmentations(fill_void='Nearest')

        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
        image2 = np.moveaxis(image2, [0, 1, 2], [1, 2, 0])

        return image, image2