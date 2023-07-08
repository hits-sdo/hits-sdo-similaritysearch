\import unittest
import os
#import dataset_aug
import numpy as np
import pyprojroot
root = pyprojroot.here()
utils_dir = root / 'search_utils'
import sys
sys.path.append(str(root))
from search_simclr.simclr.dataloader.datamodule import SimCLRDataModule



class test_data_module(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a datamodule instance
        cls.train_dir = os.path.join( root, 'data', 'miniset', 'AIA171', 'monochrome')
        cls.simclr_dm = SimCLRDataModule()

        # Call the setup() method
        cls.simclr_dm.setup('train')

    def test_train_dataloader(cls):
        cls.assertIsNotNone(cls.simclr_dm.train_dataloader)

    def test_val_dataloader(cls):
        cls.assertIsNotNone(cls.simclr_dm.val_dataloader)

    def test_test_dataloader(cls):
        cls.assertIsNotNone(cls.simclr_dm.test_dataloader)

if __name__ == "__main__":
    unittest.main()
