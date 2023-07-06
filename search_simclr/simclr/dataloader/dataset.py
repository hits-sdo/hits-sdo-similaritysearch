# Hello this is a data set class
import os, random, shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
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
    def __init__(self, tile_dir, file_list, transform=None, ):  #WHAT ARE WE USING PATH FOR?
        # self.data = data
        #self.labels = labels
        self.tile_dir = os.path.normpath(tile_dir)
        # ALL TILES MUST BE IN SAME DIRECTORY!!
        self.file_list = file_list # os.listdir(tile_dir)
        self.transform = transform # reserve transform for PyTorch transform
        # self.path = os.path.normpath(path) #/data/miniset/AIA171/monochrome/tile_20230206_000634_1024_0171_0896_0640.p 
        # full_path = f'{tile_dir}/{fname}'
    
    def __len__(self):
        # Returns length of the dataset
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Returns two images at given index
        image_fullpath = os.path.join(self.tile_dir, self.file_list[idx])
        print("image fullpath: " + image_fullpath) #/data/miniset/AIA171/monochrome/tile_20230206_000634_1024_0171_0384_0512.p
        image = image_utils.read_image(image_fullpath, 'p')
        if (self.transform):
            # Transform images by augmentations
            image1, image2 = self.transform(image)
            #lightly data set requires label, add in label later in needed
            return image1, image2, image_fullpath, image_fullpath
            
           
        else:
            # image = image_utils.read_image(image_fullpath, 'p')
            #lightly data set requires label, add in label later in needed
            
            return image, image, image_fullpath, image_fullpath
        

    ################################################################################################################### 


def fill_voids(tile_dir, file_list, image_fullpath, idx):
    # Fill voids
    image = image_utils.read_image(image_fullpath, 'p')
    v, h = image.shape[0]//2, image.shape[1]//2
    if len(image.shape) == 3:
        image = np.pad(image, ((v, v), (h, h), (0, 0)), 'edge')
    else:
        image = np.pad(image, ((v, v), (h, h)), 'edge')
        
    # Stitch images
    #image2 = image_utils.stitch_adj_imgs(tile_dir + '/', file_list[idx], file_list)
    
    # Append image (Overlay the stitched img ontop of the padded filled void image)
 
    
    return image

def partition_tile_dir_train_val(tot_file_list: List[str], 
                                 train_percent: float) -> tuple:
    '''
    Takes a list of file names and partitions the list into a
    train and validation file list
    Args: 
        tot_file_list (List(str)): Total files present in sample files directory
        train_percent (float): Percentage of data used for training model
    return: 
        list of file paths for 'train' and 'val' (tuple of lists)
    '''
    train_percentage = train_percent
    total_file_count = len(tot_file_list)
    train_file_count = int(train_percentage * total_file_count)
    train_file_list = random.sample(tot_file_list, train_file_count)
    val_file_list = [item for item in tot_file_list if item not in train_file_list]
    return train_file_list, val_file_list

def main():
    print(root)
    print("hi mom")

    test_file = os.path.join(root, 'data', 'miniset', 'AIA171', 'monochrome', 'tile_20230206_000634_1024_0171_0128_0192.p')
    image = image_utils.read_image(test_file, 'p')
    plt.imshow(image)
    plt.show()
    """  """

    tile_dir = os.path.join(root , 'data' , 'miniset' , 'AIA171' , 'monochrome')
    #tile_dir.replace(os.sep, "/")
    train_val_dir = os.path.join(root , 'data' , 'miniset' , 'AIA171' , 'train_val_simclr')
    #train_val_dir.replace(os.sep, "/")
    tot_file_list = os.listdir(tile_dir)
    train_file_list, val_file_list = partition_tile_dir_train_val(tot_file_list, 0.8)
    # save lists
    with open(os.path.join(train_val_dir,'train_file_list.txt'), 'w') as f:
        for item in train_file_list:
            f.write("%s\n" % item)
    with open(os.path.join(train_val_dir,'val_file_list.txt'), 'w') as f:
        for item in val_file_list:
            f.write("%s\n" % item)


    # Define transforms
    transform = dataset_aug.Transforms_SimCLR(blur=(1,1), 
                                              brighten=1.0, 
                                              translate=(1, 1), 
                                              zoom=1.0, 
                                              rotate=45.0, 
                                              noise_mean=0.0, 
                                              noise_std=0.05, 
                                            #   cutout_holes=0, 
                                            #   cutout_size=0,
                                            #   data_dir=tile_dir,
                                            #   file_name="tile_20230206_000634_1024_0171_0896_0640.p",
                                            #   file_list=os.listdir(tile_dir)
                                              )
    train_dataset = SdoDataset(tile_dir, train_file_list, transform=transform)
    # augmented_image1, augmented_image2 = train_dataset.__getitem__(1)
    # define a dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    # retrieve one batch of data
    for i, (augmented_image1, augmented_image2) in enumerate(train_loader):
        print(augmented_image1.shape)
        print(augmented_image2.shape)
        print(i)
        break
    
    # Plot images side-by-side
    plt.subplot(1, 2, 1)
    plt.imshow(augmented_image1[0,0,:,:].numpy())
    plt.title("image1")
    # plt.imshow(test_image.squeeze())

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image2[0,0,:,:].numpy())
    plt.title("image2")
    plt.show()

if __name__ == "__main__":
    main()
# os.listdir(path)