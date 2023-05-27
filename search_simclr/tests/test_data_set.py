import dataset
import unittest
import dataset_aug
import numpy as np
import pyprojroot
root = pyprojroot.here()
utils_dir = root / 'search_utils'
import sys
sys.path.append(str(root))

from search_simclr.dataset import SdoDataset

class test_data_set(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tile_dir = '/data/miniset/AIA171/monochrome'
        # "brighten": 1, "translate": (0, 0), "zoom": 1, "rotate": 0, "h_flip": False, "v_flip": False, 'blur': (1, 1), "p_flip": False
        augmentation_list = {"brighten": 1, "translate": (0, 0), "zoom": 1, "rotate": 0, "h_flip": False, "v_flip": False, 'blur': (1, 1), "p_flip": False}
        test_data_set = SdoDataset(tile_dir, augmentation_list, transform)
        
        
        
    def test_get_item(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    