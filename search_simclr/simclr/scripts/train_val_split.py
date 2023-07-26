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
from search_simclr.simclr.dataloader.dataset import SdoDataset, partition_tile_dir_train_val
from search_utils.file_utils import get_file_list

def main():

    print(root)

    # test_file = os.path.join(root, 'data', 'miniset', 'AIA171', 'monochrome', 'tile_20230206_000634_1024_0171_0128_0192.p')
    # image = image_utils.read_image(test_file, 'p')
    # plt.imshow(plt.show()
    """  """
    # 
    
    # data\AIA211_193_171_Miniset\20100601_000008_aia_211_193_171\tiles
    tile_dir = os.path.join(root , 'data')
    tot_fpath_wfname = os.path.join(root , 'data' , 'train_val_simclr', 'tot_full_path_files.txt')
    #tile_dir.replace(os.sep, "/")
    train_val_dir = os.path.join(root , 'data', 'train_val_simclr')
    #train_val_dir.replace(os.sep, "/")
    # Todo: FIXME !!!! Make tot_file_list a list of file full paths, not just file names
    # Todo: Use get_file_list_from_dir_recrusive() from search_utils/file_utils.py
    tot_file_list = get_file_list(tot_fpath_wfname)
    train_file_list, val_file_list = partition_tile_dir_train_val(tot_file_list[:50], 0.8)
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
                                              translate=(1, 3), 
                                              zoom=3.0, 
                                              rotate=45.0, 
                                              noise_mean=0.0, 
                                              noise_std=0.05, 
                                               cutout_holes=3, 
                                               cutout_size=0.3,
                                                data_dir=tile_dir,
                                            #   file_name="tile_20230206_000634_1024_0171_0896_0640.p",
                                                file_list=train_file_list
                                              )
    train_dataset = SdoDataset(tile_dir, train_file_list, transform=transform)
    # augmented_image1, augmented_image2 = train_dataset.__getitem__(1)
    # define a dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    # retrieve one batch of data
    for i, (augmented_image1, augmented_image2, fname, _) in enumerate(train_loader):
        print(augmented_image1.shape)
        print(augmented_image2.shape)
        print(fname)
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

if __name__== "__main__":
    main()