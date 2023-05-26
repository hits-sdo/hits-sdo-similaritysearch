# Hello this is a data set class

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import os.path
import pyprojroot
root = pyprojroot.here()
utils_dir = root / 'search_utils'
import sys
sys.path.append(str(root))
from search_utils import image_utils

# augmentations: "brighten": 1, "translate": (0, 0), "zoom": 1, "rotate": 0, "h_flip": False, "v_flip": False, 'blur': (1, 1), "p_flip": False

class SdoDataset(Dataset):
    def __init__(self, tile_dir, augmentation_list = [], transform=None):  #WHAT ARE WE USING PATH FOR?
        # self.data = data
        #self.labels = labels
        self.tile_dir = os.path.normpath(tile_dir)
        self.file_list = os.listdir(tile_dir)
        self.augmentation_list = augmentation_list # team yellow saves it as a JSON; we want to turn to dictionary
        self.transform = transform # reserve transform for PyTorch transform
        # self.path = os.path.normpath(path) #/data/miniset/AIA171/monochrome/tile_20230206_000634_1024_0171_0896_0640.p 
        # full_path = f'{tile_dir}/{fname}'
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        image_fullpath = os.path.join(self.tile_dir,self.file_list[idx])
        
        print(image_fullpath)
        image = image_utils.read_image(image_fullpath, 'p')
        return image
        # tile = open(self.tile_dir + '/' + self.file_list[idx])
        # print()

        # comment: I know in the augmentation class we have a read_image function,
        # could we just use this? This properly handles loading the contents of a
        # pickle file into an image object - Sierra
        '''
        def read_image(image_loc, image_format):
             """
            read images in pickle/jpg/png format
            and return normalized numpy array
            """
        '''
        # return tile
        

def main():
    print(root)
    print("hi mom")
    tile_dir = root / 'data' / 'miniset' / 'AIA171' / 'monochrome'
    test_dataset = SdoDataset(tile_dir)
    test_image = test_dataset.__getitem__(25)
    plt.imshow(test_image)
    plt.title("test_image")
    plt.show()
if __name__ == "__main__":
    main()
# os.listdir(path)