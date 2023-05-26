import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
import glob
from sdo_augmentation.augmentation import Augmentations
from sdo_augmentation.augmentation_list import AugmentationList
from search_utils.image_utils import read_image

class SDOTilesDataset(Dataset):
    def __init__(self, data_path: str, double_augmentation: bool):
        self.data_path = data_path
        self.image_files = glob.glob(data_path + "/**/*.jpg", recursive=True)
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

        return image, image2