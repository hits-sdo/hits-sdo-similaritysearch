import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
import glob
from search_utils.image_utils import read_image

class SDOTilesDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.image_files = glob.glob(data_path + "/**/*.jpg", recursive=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = read_image(image_loc = self.image_files[idx],
                           image_format="jpg")
        return image
