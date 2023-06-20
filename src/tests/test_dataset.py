'''
    Unit tests for database class
'''
import sys,os
sys.path.append(os.getcwd())

import unittest
import glob
import os
import numpy as np
from src.data import TilesDataset,TilesDataModule
import torch

class DatasetTest(unittest.TestCase):
    '''
        Test the data_loader class.
    '''
    def setUp(self):
        '''
            Setup the test environment.
        '''
        data_path = "data/tiles_HMI_small"
        os.path.normpath(data_path)
        image_files = glob.glob(data_path + "/**/*.npy", recursive=True)
        self.data = TilesDataModule(data_path)
        self.transform = self.data.transform
        self.database = TilesDataset(image_files, self.transform,augmentation='single')
        self.database_double_aug = TilesDataset(image_files, self.transform,augmentation='double')
        self.database_no_aug = TilesDataset(image_files, self.transform,augmentation=None)

    def test_loader_exists(self):
        '''
            Tests that the loader exists
        '''
        self.assertIsNotNone(self.database)

    def test_database_length(self):
        '''
            Tests that the database length is not 0
        '''
        print(f'Database length: {len(self.database)}')
        self.assertNotEqual(len(self.database), 0)

    def test_item_exists(self):
        '''
            Tests that __getitem__() exits
        '''
        self.assertIsNotNone(self.database[0])

    def test_item_is_tuple(self):
        '''
            Tests that the __getitem__() returns a tuple
        '''
        self.assertIsInstance(self.database[0], tuple)

    def test_images_diff(self):
        '''
            Tests that the images returned in __getitem__() are different
        '''
        image_tuple = self.database[0]
        image_tuple2 = self.database_double_aug[0]
        self.assertNotEqual(torch.sum(image_tuple[1]-image_tuple[2]), 0)
        self.assertNotEqual(torch.sum(image_tuple2[1]-image_tuple2[2]), 0)


    def test_double_augmentation(self):
        '''
            Test that the double augmentation does not return the original image
        '''
        image_tuple = self.database[0]
        image_tuple2 = self.database_double_aug[0]
        self.assertNotEqual(torch.sum(image_tuple[1]-image_tuple2[1]), 0)

    def test_no_augmentation(self):
        '''
            Test that the no augmentation returns only the original image
        '''
        image_tuple = self.database[0]
        image_tuple2 = self.database_no_aug[0]
        self.assertEqual(torch.sum(image_tuple[1]-image_tuple2[1]), 0)

    def test_dimensions(self):
        '''
            Tests that the dimensions are valid
        '''
        image_tuple = self.database[0]
        print(f'Image size: {image_tuple[1].shape}')
        self.assertEqual(image_tuple[1].shape[0], 1)
        self.assertEqual(image_tuple[2].shape[0], 1)

    def tearDown(self):
        '''
            Discards the SDO database after test methods are called
        '''
        del self.database

if __name__ == "__main__":
    unittest.main()
