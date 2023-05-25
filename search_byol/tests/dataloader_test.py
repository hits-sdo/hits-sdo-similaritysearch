# D:\Mis Documentos\AAResearch\SEARCH\Miniset\aia_171_color_1perMonth

import re
import unittest
import datetime
import os
from search_byol.data_loader import SDOTilesDataset

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

        self.sdo_dataloader = SDOTilesDataset(data_path)

    def test_loader_exists(self):
        self.assertIsNotNone(self.sdo_dataloader)

    def test_database_length(self):
        self.assertNotEqual(self.sdo_dataloader.__len__(), 0)

    def test_item_exists(self):
        self.assertIsNotNone(self.sdo_dataloader.__getitem__(0))

    def tearDown(self):
        del self.sdo_dataloader 

if __name__ == "__main__":
    unittest.main()
