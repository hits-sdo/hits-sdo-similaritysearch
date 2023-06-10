'''
    Unit tests for database class
'''

import unittest
import os
import numpy as np
from search_byol.dataset import SDOTilesDataset

class DataloaderTest(unittest.TestCase):
    '''
        Test the data_loader class.
    '''
    def setUp(self):
        '''
            Setup the test environment.
        '''
        data_path = "D:\\Mis Documentos\\AAResearch\\SEARCH\\Miniset\\aia_171_color_1perMonth"
        os.path.normpath(data_path)

        self.sdo_database = SDOTilesDataset(data_path, augmentation='single', data_stride=10)
        self.sdo_database_double_aug = SDOTilesDataset(data_path, augmentation='double', data_stride=10)
        self.sdo_database_no_aug = SDOTilesDataset(data_path, augmentation=None, data_stride=10)

    def test_loader_exists(self):
        '''
            Tests that the loader exists
        '''
        self.assertIsNotNone(self.sdo_database)

    def test_database_length(self):
        '''
            Tests that the database length is not 0
        '''
        print(f'Database length: {len(self.sdo_database)}')
        self.assertNotEqual(len(self.sdo_database), 0)

    def test_item_exists(self):
        '''
            Tests that __getitem__() exits
        '''
        self.assertIsNotNone(self.sdo_database[0])

    def test_item_is_tuple(self):
        '''
            Tests that the __getitem__() returns a tuple
        '''
        self.assertIsInstance(self.sdo_database[0], tuple)

    def test_images_diff(self):
        '''
            Tests that the images returned in __getitem__() are different
        '''
        image_tuple = self.sdo_database[0]
        image_tuple2 = self.sdo_database_double_aug[0]
        self.assertNotEqual(np.sum(image_tuple[0]-image_tuple[1]), 0)
        self.assertNotEqual(np.sum(image_tuple2[0]-image_tuple2[1]), 0)

    def test_augmentation_dict(self):
        '''
            Tests that the augmentation list is a dict
        '''
        aug_list = self.sdo_database.augmentation_list.randomize()
        self.assertIsInstance(aug_list, dict)

    def test_double_augmentation(self):
        '''
            Test that the double augmentation does not return the original image
        '''
        image_tuple = self.sdo_database[0]
        image_tuple2 = self.sdo_database_double_aug[0]
        self.assertNotEqual(np.sum(image_tuple[0]-image_tuple2[0]), 0)

    def test_no_augmentation(self):
        '''
            Test that the no augmentation returns only the original image
        '''
        image_tuple = self.sdo_database[0]
        image_tuple2 = self.sdo_database_no_aug[0]
        self.assertEqual(np.sum(image_tuple[0]-image_tuple2[0]), 0)
        self.assertEqual(image_tuple2[1],0)

    def test_dimensions(self):
        '''
            Tests that the dimensions are valid
        '''
        image_tuple = self.sdo_database[0]
        print(f'Image size: {image_tuple[0].shape}')
        self.assertEqual(image_tuple[0].shape[0], 3)
        self.assertEqual(image_tuple[1].shape[0], 3)

    def tearDown(self):
        '''
            Discards the SDO database after test methods are called
        '''
        del self.sdo_database

if __name__ == "__main__":
    unittest.main()
