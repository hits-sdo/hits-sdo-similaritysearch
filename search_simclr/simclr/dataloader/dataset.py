
# Hello this is a data set class

import torch
import numpy as np
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
from search_simclr.simclr.dataloader import dataset_aug

# augmentations: "brighten": 1, "translate": (0, 0), "zoom": 1, "rotate": 0, "h_flip": False, "v_flip": False, 'blur': (1, 1), "p_flip": False

class SdoDataset(Dataset):
    def __init__(self, tile_dir, transform=None):  #WHAT ARE WE USING PATH FOR?
        # self.data = data
        #self.labels = labels
        self.tile_dir = os.path.normpath(tile_dir)
        self.file_list = os.listdir(tile_dir)
        self.transform = transform # reserve transform for PyTorch transform
        # self.path = os.path.normpath(path) #/data/miniset/AIA171/monochrome/tile_20230206_000634_1024_0171_0896_0640.p 
        # full_path = f'{tile_dir}/{fname}'
    
    def __len__(self):
        # Returns length of the dataset
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Returns two images at given index
        image_fullpath = os.path.join(self.tile_dir, self.file_list[idx])
        
        print("Full Path: "+image_fullpath)
        
        if (self.transform):
            use_fill_voids = True
            
            if (use_fill_voids):
                # Fill voids
                image = image_utils.read_image(image_fullpath, 'p')
                v, h = image.shape[0]//2, image.shape[1]//2
                if len(image.shape) == 3:
                    image = np.pad(image, ((v, v), (h, h), (0, 0)), 'edge')
                else:
                    image = np.pad(image, ((v, v), (h, h)), 'edge')
            else:  # Stich images together
                image = image_utils.stitch_adj_imgs(self.tile_dir + '/', self.file_list[idx], self.file_list)
            
            # Transform images by augmentations
            image1, image2 = self.transform(image)
            
            # Crop middle part of image1
            shape1 = image1.shape[-2:]
            h1 = shape1[0] // 3
            h2 = shape1[0] // 3 * 2
            w1 = shape1[1] // 3
            w2 = shape1[1] // 3 * 2
            cropped_image1 = image1[:,h1:h2,w1:w2] # channels,height,width
            
            # Crop middle part of image2
            shape2 = image2.shape[-2:]
            h1 = shape2[0] // 3
            h2 = shape2[0] // 3 * 2
            w1 = shape2[1] // 3
            w2 = shape2[1] // 3 * 2
            cropped_image2 = image2[:,h1:h2,w1:w2] # channels,height,width
            
            return cropped_image1, cropped_image2
        else:
            image = image_utils.read_image(image_fullpath, 'p')
            return image, image
    
    
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
    
    # Define transforms
    transform = dataset_aug.Transforms_SimCLR(blur=(1,1), brighten=1.0, translate=(1, 1), zoom=1.0, rotate=45.0, noise_mean=0.0, noise_std=0.05, cutout_holes=0, cutout_size=0)
    test_dataset = SdoDataset(tile_dir, transform=transform)
    test_image, augmented_image = test_dataset.__getitem__(1)
    
    # Plot images side-by-side
    plt.subplot(1, 2, 1)
    plt.imshow(test_image[0,:,:].numpy())
    plt.title("image1")
    # plt.imshow(test_image.squeeze())

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image[0,:,:].numpy())
    plt.title("image2")
    plt.show()
if __name__ == "__main__":
    main()
# os.listdir(path)