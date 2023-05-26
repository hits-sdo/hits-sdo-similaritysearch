# D:\Mis Documentos\AAResearch\SEARCH\Miniset\aia_171_color_1perMonth

import re
import unittest
import datetime
import os
from search_byol.database import SDOTilesDataset
import numpy as np

class data_loader_test(unittest.TestCase):
    '''
    Test the Downloader class.
    '''
    def setUp(self):
        '''
        Setup the test environment.
        '''
        data_path = "D:\Mis Documentos\AAResearch\SEARCH\Miniset\\aia_171_color_1perMonth"
        os.path.normpath(data_path)

        self.sdo_database = SDOTilesDataset(data_path, double_augmentation=False)
        self.sdo_database_double_aug = SDOTilesDataset(data_path, double_augmentation=True)

    def test_loader_exists(self):
        self.assertIsNotNone(self.sdo_database)

    def test_database_length(self):
        self.assertNotEqual(self.sdo_database.__len__(), 0)

    def test_item_exists(self):
        self.assertIsNotNone(self.sdo_database.__getitem__(0))

    def test_item_is_tuple(self):
        self.assertIsInstance(self.sdo_database.__getitem__(0), tuple)

    def test_images_diff(self):
        image_tuple = self.sdo_database.__getitem__(0)
        self.assertNotEqual(np.sum(image_tuple[0]-image_tuple[1]), 0)

    def test_augmentation_dict(self):
        aug_list = self.sdo_database.augmentation_list.randomize()
        self.assertIsInstance(aug_list, dict)

    def test_double_augmentation(self):
        image_tuple = self.sdo_database.__getitem__(0)
        image_tuple2 = self.sdo_database_double_aug.__getitem__(0)
        self.assertNotEqual(np.sum(image_tuple[0]-image_tuple2[0]), 0)


    def tearDown(self):
        del self.sdo_database

if __name__ == "__main__":
    unittest.main()




