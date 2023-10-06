import unittest
import torch
#import dataset_aug
import numpy as np
import pyprojroot
import os
root = pyprojroot.here()
utils_dir = root / 'search_utils'
import sys
sys.path.append(str(root))
from search_simclr.simclr.dataloader.dataset import SdoDataset
from search_simclr.simclr.dataloader.dataset_aug import Transforms_SimCLR
# Transforms_SimCLR
class test_data_set(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f'root: {root}')
        # train_dir = root / 'data/miniset/AIA171/monochrome/'
        train_dir = os.path.join(root, 'data', 'miniset', 'AIA171', 'monochrome')
        file_list_txt = os.path.join(root, 'data', 'miniset', 'AIA171', 'train_val_simclr', 'train_file_list.txt') #set file list = to train list
        train_tile_list = list[str]
        # todo: open text file
        with open(file_list_txt, 'r') as file:
            train_tile_list = [line.strip() for line in file.readlines()]
        print(train_tile_list[0])

        
        transform = Transforms_SimCLR(blur=(1,1), 
                                              brighten=1.0, 
                                              translate=(1, 1), 
                                              zoom=1.0, 
                                              rotate=45.0, 
                                              noise_mean=0.0, 
                                              noise_std=0.05)
        # # "brighten": 1, "translate": (0, 0), "zoom": 1, "rotate": 0, "h_flip": False, "v_flip": False, 'blur': (1, 1), "p_flip": False
        # augmentation_list = {"brighten": 1, "translate": (0, 0), "zoom": 1, "rotate": 0, "h_flip": False, "v_flip": False, 'blur': (1, 1), "p_flip": False}
        # test_data_set = SdoDataset(tile_dir, augmentation_list, transform)

        # if test_data_set is not None:
        #     print("success!!")
        cls.train_dataset = SdoDataset(tile_dir=train_dir, file_list=train_tile_list, 
                                   transform=transform
                                        )
    
    def test_len(cls):
        cls.assertEqual(len(cls.train_dataset), 144, "The length of the dataset is inconsistent " + 
                            "with the file provided in setup")  #change to 144 files

    def test_get_item(cls):
        print(cls.train_dataset[0])
        image1, image2, image_fullpath1, image_fullpath2 = cls.train_dataset[0]
        cls.assertIsInstance(image1,torch.Tensor, "The image is not a torch tensor")
        cls.assertEqual(image1.shape, (1, 64, 64), "The shape of the image is inconsistent " + 
                            "with the file provided in setup")
        cls.assertEqual(image2.shape, (1, 64, 64), "The shape of the image is inconsistent " + 
                            "with the file provided in setup")
        cls.assertEqual(image_fullpath1, image_fullpath2, "The image full paths are inconsistent " + 
                            "with the file provided in setup")

    
    @classmethod
    def tearDownClass(cls):
        pass



if __name__ == '__main__':
    unittest.main()